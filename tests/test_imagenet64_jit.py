"""Three-rank loss equivalence and official JiT/ImageNet adapters."""
import copy
import tempfile
import json
from pathlib import Path
from PIL import Image
import unittest
import torch
from cifar_riesz_velocity import global_particle_loss
from riesz_loss import riesz_loss
from imagenet64_jit import JiTVelocity, sample_time, learning_rate, sample_velocity
from imagenet64_data import RandomImageBatches, build_index, ImageNet64

class ImageNetJiTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(51)

    def test_three_rank_gradient_matches_unpartitioned_coupled_loss(self):
        class Velocity(torch.nn.Linear):
            def forward(self, x, t):return super().forward(x)
        model=Velocity(6,6)
        clean,noise=torch.randn(10,6),torch.randn(10,6)
        # Include a shortened endpoint interval and both weighting choices.
        for time in [.1,.77,.999]:
            for time_power in [0.,.5,1.]:
                model.zero_grad()
                h=min(.01,1-time)
                source=(1-time)*noise+time*clean
                target=(1-time-h)*noise+(time+h)*clean
                expected=riesz_loss(source+h*model(source,time),target,
                    taus=(.2,.05,.02),distance_powers=(.8,1.,1.5))[0]*(1-time)**time_power/h
                expected.backward()
                reference=[p.grad.clone() for p in model.parameters()]
                h=min(.01,1-time)
                source=(1-time)*noise+time*clean
                target=(1-time-h)*noise+(time+h)*clean
                with torch.no_grad():predictions=list((source+h*model(source,time)).split([4,3,3]))
                gradients=[]
                for rank in range(3):
                    replica=copy.deepcopy(model);replica.zero_grad()
                    local=source.split([4,3,3])[rank]
                    prediction=local+h*replica(local,time)
                    loss,_=global_particle_loss(prediction,predictions,list(target.split([4,3,3])),rank,time,.01,time_power=time_power)
                    torch.testing.assert_close(loss,expected,rtol=2e-5,atol=1e-5)
                    (3*loss).backward()
                    gradients.append([p.grad.clone() for p in replica.parameters()])
                for index,wanted in enumerate(reference):
                    actual=sum(g[index] for g in gradients)/3
                    torch.testing.assert_close(actual,wanted,rtol=3e-5,atol=2e-6)

    def test_official_backbone_and_direct_rgb_velocity(self):
        model=JiTVelocity()
        self.assertEqual(sum(p.numel() for p in model.parameters()),131136000)
        self.assertEqual(model.backbone.in_context_len,32)
        self.assertEqual(model.backbone.in_context_start,4)
        self.assertEqual(len(model.backbone.blocks),12)
        torch.nn.init.normal_(model.backbone.final_layer.linear.weight,std=.01)
        x=torch.randn(2,3,64,64)
        t=torch.tensor(.35)
        actual=model(x,t)
        expected=model.backbone(x,t.expand(2),torch.full((2,),1000))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all()
                            for p in model.parameters() if p.requires_grad))

    def test_jit_time_distribution_and_warmup(self):
        rng=torch.Generator().manual_seed(71)
        times=torch.stack([sample_time('cpu',generator=rng) for _ in range(12000)])
        self.assertAlmostEqual(float(times.median()),float(torch.sigmoid(torch.tensor(-.8))),delta=.015)
        self.assertTrue(((times>=.0001)&(times<=.999)).all())
        peak=5e-5*192/256
        self.assertEqual(learning_rate(4999,peak,5000),peak)
        self.assertEqual(learning_rate(50000,peak,5000),peak)
        self.assertLess(learning_rate(0,peak,5000),peak)

    def test_data_attempt_resume_is_prefetch_independent(self):
        images=torch.arange(80).reshape(20,1,2,2)
        batches=RandomImageBatches(images,8,71)
        expected=batches[13]["images"]
        for i in range(14,21):batches[i]
        resumed=RandomImageBatches(images,8,71)
        torch.testing.assert_close(resumed[13]["images"],expected,rtol=0,atol=0)

    def test_cfg_is_applied_only_inside_interval(self):
        class LabelVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.null_times=[]
            def forward(self,x,t,labels):
                if int(labels[0])==1000:self.null_times.append(float(t))
                return torch.ones_like(x)*(labels!=1000)[:,None,None,None]
        model=LabelVelocity()
        x=torch.zeros(2,3,2,2)
        actual=sample_velocity(model,x,10,False,'euler',torch.zeros(2,dtype=torch.long),2.9)
        torch.testing.assert_close(actual,torch.full_like(x,2.71))
        self.assertEqual(len(model.null_times),9)
        self.assertTrue(all(.09999<=t<=1 for t in model.null_times))
        model.null_times=[]
        sample_velocity(model,x,50,False,'heun',torch.zeros(2,dtype=torch.long),2.9)
        self.assertAlmostEqual(max(model.null_times),1.)
        self.assertTrue(all(.09999<=t<=1 for t in model.null_times))

    def test_class_batches_and_dropout_agree_across_ranks(self):
        class Images:
            records=[(str(i),i) for i in range(1000)]
            def __len__(self):return 1000
            def __getitem__(self,i):return torch.tensor([i])
        data=Images()
        a=RandomImageBatches(data,4,7,0,.1)
        b=RandomImageBatches(data,3,7,1,.1)
        dropped=0
        for attempt in range(1000):
            first,second=a[attempt],b[attempt]
            self.assertEqual(first['label'],second['label'])
            if first['label']==1000:
                dropped+=1
            else:
                self.assertTrue((first['images']==first['label']).all())
                self.assertTrue((second['images']==second['label']).all())
        self.assertTrue(60<dropped<140)

    def test_manifest_index_matches_real_rgb_image_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            root=base/'ILSVRC/Data/CLS-LOC/train'
            output=base/'output';output.mkdir()
            rows=['ImageId,PredictionString']
            for name in ['n00000001','n00000002']:
                (root/name).mkdir(parents=True)
                for i in range(2):
                    image_id=f'{name}_{i}'
                    Image.new('RGB',(80,100)).save(root/name/f'{image_id}.JPEG')
                    rows.append(f'{image_id},{name} 0 0 1 1')
            (base/'LOC_train_solution.csv').write_text('\n'.join(rows)+'\n')
            identity=build_index(root,output,expected_count=4,expected_classes=2)
            self.assertEqual(identity['count'],4)
            images=ImageNet64(root,output/'dataset_index.json')
            self.assertEqual(tuple(images[0].shape),(3,64,64))
            self.assertEqual(images[0].dtype,torch.uint8)
            self.assertEqual(len(identity['spot_sha256']),4)
            (base/'LOC_train_solution.csv').write_text('\n'.join(rows[:2])+'\n')
            complete=build_index(root,output,expected_count=4,expected_classes=2)
            self.assertEqual(complete['count'],4)
            self.assertEqual(complete['index_sha256'],identity['index_sha256'])

if __name__=='__main__':unittest.main()
