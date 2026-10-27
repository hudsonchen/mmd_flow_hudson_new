"""Small time-conditioned models and reference-path transport experiments."""
import math
import torch
from backbone import ResidualBlock, SpatialAttention
from loss import _riesz_topk_softmax_loss_impl


def validate_riesz_parent(checkpoint, _seen=None):
    """Reject direct or inherited FM weights; older checkpoints require ancestry audit."""
    if checkpoint.get('lineage_has_flow_matching',False) or checkpoint['config']['objective'] in ['fm','flow-matching']:
        raise ValueError('Riesz runs must not inherit flow-matching weights')
    if checkpoint['config'].get('dit_prime_checkpoint'):
        from reference_dit import verified_reconstruction_checkpoint
        verified_reconstruction_checkpoint(checkpoint['config']['dit_prime_checkpoint'],
            checkpoint['config'].get('dit_priming_sha256'))
    if checkpoint['config'].get('fork'):
        from pathlib import Path
        parent=checkpoint['config']['fork']
        path=str(Path(parent).resolve())
        seen=set() if _seen is None else _seen
        if path in seen: raise ValueError('Cyclic checkpoint ancestry')
        seen.add(path)
        validate_riesz_parent(torch.load(parent,map_location='cpu',weights_only=False),seen)


def verify_reconstruction_encoder(checkpoint):
    """Verify a frozen encoder against its actual reconstruction-only origin."""
    import hashlib
    from pathlib import Path
    encoder=checkpoint.get('encoder')
    if encoder is None: return dict(present=False)
    ancestor=checkpoint
    ancestor_path=None
    seen=set()
    while True:
        config=ancestor['config']
        if config.get('prime_steps',0) or (config.get('objective')=='reconstruction' and ancestor_path is not None):
            origin=Path(config['output'])/'reconstruction.pt' if config.get('prime_steps',0) else ancestor_path
            payload=origin.read_bytes()
            import io
            reconstruction=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
            if reconstruction['config']['objective']!='reconstruction':
                raise ValueError('Encoder origin is not reconstruction priming')
            validate_riesz_parent(reconstruction)
            original=reconstruction['model']
            if set(encoder)!=set(original) or any(not torch.equal(encoder[k].detach().cpu(),original[k].detach().cpu()) for k in original):
                raise ValueError('Frozen encoder differs from its reconstruction-priming weights')
            return dict(present=True,unchanged=True,origin=str(origin),
                sha256=hashlib.sha256(payload).hexdigest(),priming_updates=reconstruction['update'])
        parent=config.get('fork')
        if not parent: raise ValueError('Cannot locate reconstruction origin for saved encoder')
        path=str(Path(parent).resolve())
        if path in seen: raise ValueError('Cyclic encoder ancestry')
        seen.add(path)
        ancestor_path=Path(parent)
        ancestor=torch.load(parent,map_location='cpu',weights_only=False)


def restore_image_checkpoint(model,checkpoint,encoder=None,keep_encoder=False,require_riesz=True):
    """Restore generator EMA while optionally retaining a newly primed encoder."""
    if require_riesz: validate_riesz_parent(checkpoint)
    if keep_encoder and encoder is None:
        raise ValueError('Keeping a new encoder requires a primed encoder')
    if encoder is not None and any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('The feature encoder must be frozen before restoration')
    model.load_state_dict(checkpoint['ema'])
    if encoder is not None and not keep_encoder:
        state=checkpoint.get('encoder')
        if state is None and checkpoint['config']['objective']=='reconstruction':
            state=checkpoint['model']
        if state is None: raise ValueError('Parent checkpoint has no frozen encoder')
        encoder.load_state_dict(state)


def time_features(t, x, width=32):
    t = torch.as_tensor(t, device=x.device, dtype=x.dtype).expand(len(x))
    frequencies = torch.exp(torch.linspace(0, math.log(1000), width // 2, device=x.device))
    phase = t[:, None] * frequencies
    return torch.cat([phase.sin(), phase.cos()], -1)


class ToyVelocity(torch.nn.Module):
    def __init__(self, width=128, spatial_fourier=False):
        super().__init__()
        frequencies=(torch.randn(2,64)*torch.tensor([.5,1.,2.,4.]).repeat_interleave(16)
                     if spatial_fourier else torch.empty(2,0))
        self.register_buffer('frequencies',frequencies,persistent=spatial_fourier)
        self.net = torch.nn.Sequential(torch.nn.Linear(34+2*frequencies.shape[1], width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 2))

    def forward(self, x, t):
        features=[x,time_features(t,x)]
        if self.frequencies.numel():
            phase=x@self.frequencies
            features += [phase.sin(),phase.cos()]
        return self.net(torch.cat(features,-1))


class PotentialVelocity(ToyVelocity):
    """Constrain local transport to a gradient field instead of arbitrary rotations."""
    def __init__(self,width=128,spatial_fourier=False):
        super().__init__(width,spatial_fourier)
        self.net[-1]=torch.nn.Linear(width,1)

    def forward(self,x,t):
        differentiable=torch.is_grad_enabled()
        with torch.enable_grad():
            state=x if x.requires_grad else x.detach().requires_grad_(True)
            potential=super().forward(state,t)
            velocity=torch.autograd.grad(potential.sum(),state,create_graph=differentiable)[0]
        return velocity


def image_geometry(dataset):
    if dataset in ['mnist','fashionmnist']: return (1,28,28)
    if dataset=='cifar10': return (3,32,32)
    raise ValueError(f'Unknown image dataset: {dataset}')


class ImageVelocity(torch.nn.Module):
    """Small image U-Net, unrestricted velocity, per-block time conditioning."""
    def __init__(self, width=32, spatial_fourier=False, deep=False, channels=1, image_size=28):
        super().__init__()
        if channels<1 or image_size<4 or image_size%4:
            raise ValueError('Positive channels and image size divisible by four are required')
        self.image_shape=(channels,image_size,image_size)
        self.spatial_fourier=spatial_fourier
        self.deep=deep
        self.time = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128))
        self.input = torch.nn.Conv2d(channels*(9 if spatial_fourier else 1), width, 3, padding=1)
        self.enc = ResidualBlock(width, time_channels=128, dropout=0)
        self.down = ResidualBlock(width, width*2, down=True, time_channels=128, dropout=0)
        middle=width*4 if deep else width*2
        if deep:
            self.down7=ResidualBlock(width*2,middle,down=True,time_channels=128,dropout=0)
            self.attn7=SpatialAttention(middle)
            self.up14=ResidualBlock(middle,width*2,up=True,time_channels=128,dropout=0)
            self.dec14=ResidualBlock(width*4,width*2,time_channels=128,dropout=0)
        self.mid1 = ResidualBlock(middle, middle, time_channels=128, dropout=0)
        self.mid2 = ResidualBlock(middle, middle, time_channels=128, dropout=0)
        self.up = ResidualBlock(width*2, width, up=True, time_channels=128, dropout=0)
        self.dec = ResidualBlock(width*2, width, time_channels=128, dropout=0)
        self.output = torch.nn.Conv2d(width, channels, 3, padding=1)

    def _encode(self,x,t):
        emb = self.time(time_features(t, x))
        image=x.reshape(-1,*self.image_shape)
        if self.spatial_fourier:
            phase=(image[:,:,None]*image.new_tensor([.5,1.,2.,4.])[None,None,:,None,None]).flatten(1,2)
            image=torch.cat([image,phase.sin(),phase.cos()],1)
        skip = self.enc(self.input(image), emb)
        low=self.down(skip,emb)
        h=self.down7(low,emb) if self.deep else low
        h=self.mid2(self.mid1(h,emb),emb)
        if self.deep: h=self.attn7(h)
        return emb,skip,low,h

    def forward(self,x,t):
        emb,skip,low,h=self._encode(x,t)
        if self.deep: h=self.dec14(torch.cat([self.up14(h,emb),low],1),emb)
        return self.output(self.dec(torch.cat([self.up(h, emb), skip], 1), emb)).flatten(1)

    def features(self,x):
        return [torch.nn.functional.adaptive_avg_pool2d(f,(4,4)).flatten(1) for f in self.feature_maps(x)]

    def feature_maps(self,x):
        _,skip,low,h=self._encode(x,1.)
        return [skip,low,h] if self.deep else [skip,h]


def sample(model, noise, steps=50, method='heun', checkpoint_steps=False):
    from torch.utils.checkpoint import checkpoint
    x = noise
    for k in range(steps):
        t, h = k / steps, 1 / steps
        def advance(state, time=t):
            v = model(state, time)
            next_x = state + h*v
            return (state + h/2*(v + model(next_x, time+h))) if method == 'heun' else next_x
        x = checkpoint(advance,x,use_reentrant=False) if checkpoint_steps else advance(x)
    return x


class DenoiserVelocity(torch.nn.Module):
    """Parameterize the affine reference flow through a clean-state predictor.

    v(x,t)=(D(x,t)-x)/(1-t). An Euler rollout's last step directly emits D;
    the endpoint is not a residual correction added to accumulated state noise.
    The floor only defines the singular t=1 evaluation (Euler avoids it).
    """
    def __init__(self,base):
        super().__init__()
        self.base=base

    def reconstruct(self,x,t):
        return self.base(x,t)

    def forward(self,x,t):
        time=torch.as_tensor(t,device=x.device,dtype=x.dtype).expand(len(x))
        remaining=(1-time).clamp_min(.001).reshape(-1,1)
        return (self.base(x,time)-x)/remaining

    def features(self,x):
        return self.base.features(x)

    def feature_maps(self,x):
        return self.base.feature_maps(x)


def row_column_riesz_loss(prediction,target,taus,normalization,negative,repulsion_weight,exclude_target_diagonal=False,
                          query_tail_scale=0.,query_tail_max_weight=4.,include_self_negative=False,distance_smoothing=0.):
    """Drifting row/column affinities with Riesz unit-direction forces.

    Positive and negative affinities share softmax normalization. Cross-mass
    balancing follows Drifting, but forces use unit directions divided by tau,
    rather than Drifting's raw displacements. Targets and fields are detached.
    """
    if len(prediction)<2 or len(target)<1: raise ValueError('Need at least two queries and one target')
    if not taus or any(t<=0 for t in taus): raise ValueError('Temperatures must be positive')
    if normalization not in ['none','temperature','global']: raise ValueError('Unknown field normalization')
    if not math.isfinite(query_tail_scale) or query_tail_scale<0 or not math.isfinite(query_tail_max_weight) or query_tail_max_weight<1:
        raise ValueError('Invalid query-tail weighting settings')
    if exclude_target_diagonal and len(prediction)!=len(target):
        raise ValueError('Matched-target exclusion requires equally sized aligned populations')
    if not math.isfinite(distance_smoothing) or distance_smoothing<0:
        raise ValueError('Distance smoothing must be finite and nonnegative')
    if include_self_negative and (negative is not None or exclude_target_diagonal):
        raise ValueError('Self-negative normalization requires generated negatives and included positive diagonal')
    with torch.no_grad():
        query=prediction.detach().float()
        negatives=query if negative is None else negative.detach().float()
        positive=target.detach().float()
        support=torch.cat([negatives,positive])
        distances=torch.cdist(query,support,compute_mode='donot_use_mm_for_euclid_dist')
        scale=distances.mean().clamp_min(1e-3)
        scale_inputs=(scale/math.sqrt(query.shape[-1])).clamp_min(1e-3)
        smoothing=distance_smoothing*scale
        # Smooth norm potential sqrt(r^2+eps^2)-eps, with eps relative to
        # the batch distance scale. Its force vanishes continuously at r=0.
        normalized_distance=((distances.square()+smoothing.square()).sqrt()-smoothing)/scale if distance_smoothing else distances/scale
        mask=torch.zeros_like(distances,dtype=torch.bool)
        if negative is None and not include_self_negative:
            mask[:,:len(query)]=torch.eye(len(query),device=query.device,dtype=torch.bool)
        # A self negative has zero displacement, but may participate in affinity
        # normalization. Including it restores exact cancellation when the two
        # empirical populations coincide and positive self matches are included.
        split=len(negatives)
        if exclude_target_diagonal:
            mask[:,split:]=torch.eye(len(query),device=query.device,dtype=torch.bool)
        query_weights=torch.ones(len(query),1,device=query.device)
        if query_tail_scale:
            nearest=normalized_distance[:,split:].masked_fill(mask[:,split:],float('inf')).min(-1).values
            query_weights=(1+(nearest/query_tail_scale).square()).clamp_max(query_tail_max_weight)[:,None]
        field=torch.zeros_like(query)
        for tau in taus:
            logits=(-normalized_distance/tau).masked_fill(mask,float('-inf'))
            affinity=(logits.softmax(-1)*logits.softmax(-2)).sqrt()
            neg,pos=affinity[:,:split],affinity[:,split:]
            pos_coeff=pos*neg.sum(-1,keepdim=True)
            neg_coeff=neg*pos.sum(-1,keepdim=True)
            chunks=[]
            for start in range(0,len(query),32):
                q=query[start:start+32]
                dp=positive[None]-q[:,None]
                dn=negatives[None]-q[:,None]
                positive_norm=(dp.square().sum(-1)+smoothing.square()).sqrt() if distance_smoothing else dp.norm(dim=-1)
                negative_norm=(dn.square().sum(-1)+smoothing.square()).sqrt() if distance_smoothing else dn.norm(dim=-1)
                attraction=(dp*(pos_coeff[start:start+32]/positive_norm.clamp_min(1e-6))[...,None]).sum(1)
                repulsion=(dn*(neg_coeff[start:start+32]/negative_norm.clamp_min(1e-6))[...,None]).sum(1)
                chunks.append((attraction-repulsion_weight*repulsion)/tau)
            force=torch.cat(chunks)
            if query_tail_scale:force=force*query_weights
            if normalization=='temperature': force=force/force.square().mean().clamp_min(1e-8).sqrt()
            field+=force
        if normalization=='global': field=field/field.square().mean().sqrt().clamp_min(1e-8)
        goal=query/scale_inputs+field
    return (prediction/scale_inputs-goal).square().mean(),dict(frozen_velocity=field,scale_inputs=scale_inputs,query_tail_weights=query_weights)


def riesz_loss(prediction, target, taus=(.1,), normalization='none', negative=None, repulsion_weight=1.,affinity='row',exclude_target_diagonal=False,
               query_tail_scale=0.,query_tail_max_weight=4.,include_self_negative=False,distance_smoothing=0.):
    if affinity=='row-column':
        return row_column_riesz_loss(prediction,target,taus,normalization,negative,repulsion_weight,exclude_target_diagonal,
            query_tail_scale,query_tail_max_weight,include_self_negative,distance_smoothing)
    if affinity!='row': raise ValueError('Unknown Riesz affinity')
    if include_self_negative or distance_smoothing:raise ValueError('Self normalization and smoothing require row-column affinity')
    if exclude_target_diagonal: raise ValueError('Matched-target exclusion requires row-column affinity')
    if query_tail_scale:raise ValueError('Query-tail weighting requires row-column affinity')
    value, info = _riesz_topk_softmax_loss_impl(prediction[None], target[None],
        fixed_neg=negative[None] if negative is not None else None,
        weight_gen=torch.zeros_like(prediction[None,:,0]) if negative is not None else None,
        tau_list=taus, no_field_rms_normalization=normalization=='none',
        normalize_per_temperature=normalization=='temperature', repulsion_weight=repulsion_weight, epsilon=1e-6)
    return value.mean(), info


def marginal_loss(model, clean, noise, target, target_noise, t, horizon, coordinate_scale=None, time_weight_power=0.,
                  pixel_scales=None, image_shape=None, **kwargs):
    """Independent target marginal; divide by h to avoid vanishing velocity gradients."""
    h = min(horizon, 1-float(t))
    source = (1-t)*noise + t*clean
    future = (1-t-h)*target_noise + (t+h)*target
    prediction=source+h*model(source,t)
    if coordinate_scale is not None:
        prediction,future=prediction/coordinate_scale,future/coordinate_scale
    if not math.isfinite(time_weight_power) or time_weight_power<0:
        raise ValueError('Reference time power must be finite and nonnegative')
    # For DenoiserVelocity, power=1 cancels the explicit 1/(1-t) Jacobian.
    # This is a loss weighting, not a change to the reference distributions.
    if pixel_scales is None:
        value=riesz_loss(prediction, future, **kwargs)[0]
    else:
        if (image_shape is None or len(image_shape)!=3 or math.prod(image_shape)!=prediction.shape[-1]
                or not pixel_scales or len(set(pixel_scales))!=len(pixel_scales)
                or any(side<1 or side>min(image_shape[-2:]) for side in pixel_scales)):
            raise ValueError('Pixel scales require a matching image shape and unique valid resolutions')
        # Fixed average pooling only: each resolution has its own Riesz field.
        # Model evaluation and the reference time are shared across resolutions.
        generated=prediction.reshape(-1,*image_shape)
        reference=future.reshape(-1,*image_shape)
        values=[]
        for side in pixel_scales:
            a=torch.nn.functional.adaptive_avg_pool2d(generated,(side,side)).flatten(1)
            b=torch.nn.functional.adaptive_avg_pool2d(reference,(side,side)).flatten(1)
            values.append(riesz_loss(a,b,**kwargs)[0])
        value=torch.stack(values).mean()
    return (1-float(t))**time_weight_power*value / h


def coupled_reference_loss(model, clean, noise, t, horizon, **kwargs):
    """Common random particles for adjacent reference marginals.

    This intentionally correlates source and target Monte Carlo populations.
    Soft neighbor attraction still sees every target, and self repulsion excludes
    the query. It is not the independent-population estimator used above.
    """
    return marginal_loss(model,clean,noise,clean,noise,t,horizon,**kwargs)


def local_reference_hybrid_loss(model, clean, noise, target, target_noise, t, horizon,
                               reference_weight=1.,riesz_weight=.1,time_weight_power=1.,
                               coordinate_scale=None,**kwargs):
    """One-call hybrid: paired reference denoising plus next-marginal Riesz.

    Clean-image MSE equals (1-t)^2 times conditional velocity MSE for the affine
    path. This is joint supervised/distributional training, not pure Riesz.
    It uses analytic data/noise pairs, no trained FM teacher or rollout graph.
    Target population coupling is explicit at the caller.
    """
    if not isinstance(model,DenoiserVelocity):raise ValueError('Local hybrid requires a clean-image predictor')
    if not 0<=float(t)<1 or not 0<float(horizon)<=1:raise ValueError('Invalid local hybrid time or horizon')
    if any(not math.isfinite(w) or w<=0 for w in [reference_weight,riesz_weight]):
        raise ValueError('Both hybrid loss weights must be finite and positive')
    if not math.isfinite(time_weight_power) or time_weight_power<0:raise ValueError('Invalid time weight')
    clean,noise,target,target_noise=(x.detach() for x in [clean,noise,target,target_noise])
    h=min(float(horizon),1-float(t))
    source=(1-t)*noise+t*clean
    denoised=model.reconstruct(source,t)
    # Convex form avoids cancellation at the final interval, where alpha=1.
    alpha=h/max(1-float(t),.001)
    prediction=(1-alpha)*source+alpha*denoised
    future=(1-t-h)*target_noise+(t+h)*target
    if coordinate_scale is not None:prediction,future=prediction/coordinate_scale,future/coordinate_scale
    local=(1-float(t))**time_weight_power*riesz_loss(prediction,future,**kwargs)[0]/h
    reference=(denoised-clean).square().mean()
    return reference_weight*reference+riesz_weight*local,dict(
        reference_clean_mse=reference.detach(),local_riesz=local.detach(),
        weighted_reference_clean_mse=reference_weight*reference.detach(),weighted_local_riesz=riesz_weight*local.detach())


def pixel_moment_features(images,image_shape=(1,28,28)):
    """Aligned local means and standard deviations; no image pairing or labels."""
    x=images.reshape(-1,*image_shape)
    features=[]
    for side in (2,4):
        patches=torch.nn.functional.unfold(x,side,stride=side)
        # Keep RGB channels separate: color differences are not spatial variance.
        patches=patches.reshape(len(x),image_shape[0],side*side,-1)
        mean=patches.mean(2)
        # Center explicitly: E[x^2]-E[x]^2 loses near-black small variances.
        variance=(patches-mean[:,:,None]).square().mean(2)
        features.extend([mean.flatten(1),(variance+1e-8).sqrt().flatten(1)])
    return features


def pixel_gradient_features(images,image_shape=(1,28,28)):
    """Signed spatial derivatives at multiple scales; invariant to brightness shifts."""
    x=images.reshape(-1,*image_shape)
    dx=x[:,:,:,1:]-x[:,:,:,:-1]
    dy=x[:,:,1:,:]-x[:,:,:-1,:]
    return [torch.cat([dx.flatten(1),dy.flatten(1)],1)]+[
            torch.cat([torch.nn.functional.adaptive_avg_pool2d(dx,(side,side)).flatten(1),
                       torch.nn.functional.adaptive_avg_pool2d(dy,(side,side)).flatten(1)],1)
            for side in (image_shape[-1]//2,image_shape[-1]//4)]


def spatial_riesz_loss(prediction,target,taus=(.1,),normalization='none',repulsion_weight=1.,affinity='row-column'):
    """Independent location kernels, shared map normalization; inputs [L,N,C].

    Query chunks bound the memory for explicit unit-direction displacements,
    avoiding cancellation in weighted coordinate sums for near-collisions.
    This is an unpaired endpoint loss; only self negatives are masked.
    """
    if affinity!='row-column': raise ValueError('Spatial Riesz requires row-column affinities')
    if prediction.ndim!=3 or target.ndim!=3 or prediction.shape[::2]!=target.shape[::2]:
        raise ValueError('Spatial features must have matching location and channel dimensions')
    if prediction.shape[1]<2 or target.shape[1]<1: raise ValueError('Insufficient spatial particles')
    if not taus or any(t<=0 for t in taus): raise ValueError('Temperatures must be positive')
    if normalization not in ['none','temperature','global']: raise ValueError('Unknown field normalization')
    with torch.no_grad():
        q=prediction.detach().float()
        positive=target.detach().float()
        n=q.shape[1]
        support=torch.cat([q,positive],1)
        distances=torch.cdist(q,support,compute_mode='donot_use_mm_for_euclid_dist')
        scale=distances.mean().clamp_min(1e-3)
        scale_inputs=(scale/math.sqrt(q.shape[-1])).clamp_min(1e-3)
        mask=torch.zeros_like(distances,dtype=torch.bool)
        mask[:,:,:n]=torch.eye(n,device=q.device,dtype=torch.bool)
        field=torch.zeros_like(q)
        for tau in taus:
            logits=(-distances/scale/tau).masked_fill(mask,float('-inf'))
            affinities=(logits.softmax(-1)*logits.softmax(-2)).sqrt()
            neg,pos=affinities[:,:,:n],affinities[:,:,n:]
            pc=pos*neg.sum(-1,keepdim=True)
            nc=neg*pos.sum(-1,keepdim=True)
            chunks=[]
            for start in range(0,n,16):
                query=q[:,start:start+16]
                dp=positive[:,None]-query[:,:,None]
                dn=q[:,None]-query[:,:,None]
                attraction=(dp*(pc[:,start:start+16]/dp.norm(dim=-1).clamp_min(1e-6))[...,None]).sum(2)
                repulsion=(dn*(nc[:,start:start+16]/dn.norm(dim=-1).clamp_min(1e-6))[...,None]).sum(2)
                chunks.append((attraction-repulsion_weight*repulsion)/tau)
            force=torch.cat(chunks,1)
            if normalization=='temperature': force=force/force.square().mean().clamp_min(1e-8).sqrt()
            field+=force
        if normalization=='global': field=field/field.square().mean().sqrt().clamp_min(1e-8)
        goal=q/scale_inputs+field
    return (prediction/scale_inputs-goal).square().mean()


def image_riesz_loss(prediction,target,encoder=None,statistics_weight=0.,gradient_weight=0.,pixel_scale=None,
                     spatial_feature_weight=0.,spatial_feature_locations=16,image_shape=(1,28,28),**kwargs):
    """Equal-weight pixel and clean-image reconstruction-feature fields."""
    pixel_prediction=prediction if pixel_scale is None else prediction/pixel_scale
    pixel_target=target if pixel_scale is None else target/pixel_scale
    values=[riesz_loss(pixel_prediction,pixel_target,**kwargs)[0]]
    if encoder is not None:
        if spatial_feature_weight:
            generated_maps=encoder.feature_maps(prediction)
            with torch.no_grad(): target_maps=encoder.feature_maps(target)
            generated_features=[torch.nn.functional.adaptive_avg_pool2d(f,(4,4)).flatten(1) for f in generated_maps]
            target_features=[torch.nn.functional.adaptive_avg_pool2d(f,(4,4)).flatten(1) for f in target_maps]
        else:
            generated_features=encoder.features(prediction)
            with torch.no_grad(): target_features=encoder.features(target)
        values += [riesz_loss(a,b,**kwargs)[0] for a,b in zip(generated_features,target_features)]
    total=torch.stack(values).mean()
    weight=1.
    if spatial_feature_weight:
        if encoder is None: raise ValueError('Spatial features require a frozen reconstruction encoder')
        if spatial_feature_locations<1: raise ValueError('At least one spatial location is required')
        spatial=[]
        for a,b in zip(generated_maps,target_maps):
            a,b=a.flatten(2),b.flatten(2)
            locations=torch.randperm(a.shape[-1],device=a.device)[:spatial_feature_locations]
            spatial.append(spatial_riesz_loss(a[:,:,locations].permute(2,0,1),b[:,:,locations].permute(2,0,1),**kwargs))
            for transform in (lambda x:x.mean(-1),lambda x:((x-x.mean(-1,keepdim=True)).square().mean(-1)+1e-8).sqrt()):
                spatial.append(spatial_riesz_loss(transform(a)[None],transform(b)[None],**kwargs))
        total=total+spatial_feature_weight*torch.stack(spatial).mean()
        weight+=spatial_feature_weight
    if statistics_weight:
        moments=pixel_moment_features(prediction,image_shape)
        with torch.no_grad(): target_moments=pixel_moment_features(target,image_shape)
        statistics=torch.stack([riesz_loss(x,y,**kwargs)[0] for x,y in zip(moments,target_moments)]).mean()
        total=total+statistics_weight*statistics
        weight+=statistics_weight
    if gradient_weight:
        gradients=pixel_gradient_features(prediction,image_shape)
        with torch.no_grad(): target_gradients=pixel_gradient_features(target,image_shape)
        gradient_loss=torch.stack([riesz_loss(x,y,**kwargs)[0] for x,y in zip(gradients,target_gradients)]).mean()
        total=total+gradient_weight*gradient_loss
        weight+=gradient_weight
    return total/weight


def vgg_endpoint_riesz_loss(prediction,target,encoder,pool_size=8,**kwargs):
    """Unpaired Riesz fields over frozen VGG spatial/local/global feature branches.

    Inputs are centered RGB image states. The affine conversion to [0,1] is
    deliberately unclipped, retaining gradients for out-of-range predictions.
    The VGG backend applies its normal input scaling. No LPIPS pairwise heads,
    paired reconstruction targets, or reference-path VGG supervision are used.
    """
    if prediction.ndim!=2 or target.ndim!=2 or prediction.shape[1]!=3072 or target.shape[1]!=3072:
        raise ValueError('VGG endpoint fields require flattened RGB32 images')
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('VGG endpoint encoder must be frozen')
    generated=encoder.multiscale_features((prediction+1)/2,pool_size=pool_size)
    with torch.no_grad():
        real=encoder.multiscale_features((target.detach()+1)/2,pool_size=pool_size)
    if not generated or generated.keys()!=real.keys():
        raise ValueError('VGG feature branches must match')
    return torch.stack([riesz_loss(value,real[name],**kwargs)[0] for name,value in generated.items()]).mean()


def reference_centered_loss(model,clean,noise,target,target_noise,negative_clean,negative_noise,
                            t,horizon,taus=(.1,),normalization='none'):
    """Reference-field control variate with equal-sized supports.

    The query is independent of both supports. Supports may be independent or
    paired (the same clean/noise particles); paired supports cancel the baseline
    exactly and reduce sampling variance. Each population
    reuses its own random path particles across t and t+h. At t both support
    populations have exactly the same law, so swapping them negates the baseline
    field: its expectation is zero when repulsion weight is one. Subtract the
    matched baseline field before regressing the velocity. Independent negative
    model outputs are detached, as in particle-field training. No teacher model
    or paired velocity target is used.
    """
    if len(target)!=len(negative_clean):
        raise ValueError('Reference control variate requires equal support sizes')
    h=min(horizon,1-float(t))
    if h<=0: raise ValueError('Reference control variate requires positive horizon')
    # Increment form makes a stationary path cancel exactly, including in fp32.
    source=noise+t*(clean-noise)
    positive=target_noise+t*(target-target_noise)
    negative=negative_noise+t*(negative_clean-negative_noise)
    velocity=model(source,t)
    prediction=source+h*velocity
    with torch.no_grad():
        future_positive=positive+h*(target-target_noise)
        future_negative=negative+h*model(negative,t)
        options=dict(tau_list=taus,no_field_rms_normalization=normalization=='none',
            normalize_per_temperature=normalization=='temperature',return_field=True,epsilon=1e-6)
        def force(query,pos,neg):
            _,info=_riesz_topk_softmax_loss_impl(query[None],pos[None],fixed_neg=neg[None],
                weight_gen=torch.zeros_like(query[None,:,0]),**options)
            return info['frozen_velocity'][0]/info['scale_inputs']
        baseline=force(source,positive,negative)
        future=force(prediction.detach(),future_positive,future_negative)
        corrected=(future-baseline)/h
        goal=velocity.detach()+corrected
    return (velocity-goal).square().mean()
