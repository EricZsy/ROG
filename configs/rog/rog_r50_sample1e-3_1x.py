_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='ROGShared2FCBBoxHead',
            num_classes=1203,
            gamma=1.0,
            lam=0.1,
            ),
        mask_head=dict(num_classes=1203,
                       )),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2)) # detectron2 default.
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(dataset=dict(pipeline=train_pipeline))
)
evaluation = dict(interval=12, metric=['bbox', 'segm'])

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

work_dir = 'work_dirs/rog_r50_sample1e-3_1x'