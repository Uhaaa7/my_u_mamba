import torch
import torch.nn as nn
from UMambaBot_2d import UMambaBot

def test():
    print("🚀 正在初始化 UMambaBot (DCNv2 + H-SS2D)...")
    
    # 1. 准备必要的配置参数 (修复报错的关键)
    norm_op = nn.InstanceNorm2d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op = None
    dropout_op_kwargs = None
    nonlin = nn.LeakyReLU
    nonlin_kwargs = {'inplace': True}

    # 2. 实例化模型 (模拟真实训练时的参数)
    model = UMambaBot(
        input_channels=1, 
        n_stages=4, 
        features_per_stage=[32, 64, 128, 256],
        conv_op=nn.Conv2d, 
        kernel_sizes=[[3,3]]*4, 
        strides=[[1,1],[2,2],[2,2],[2,2]],
        n_conv_per_stage=[2,2,2,2], 
        num_classes=2, 
        n_conv_per_stage_decoder=[2,2,2],
        deep_supervision=True,  # 开启深监督
        
        # === 关键参数 ===
        norm_op=norm_op,
        norm_op_kwargs=norm_op_kwargs,
        dropout_op=dropout_op,
        dropout_op_kwargs=dropout_op_kwargs,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs
    ).cuda()

    # 3. 构造虚拟输入
    # Batch=2 以测试 Batch Norm/Instance Norm 的行为
    x = torch.randn(2, 1, 128, 128).cuda()

    print("🌊 开始前向传播测试...")
    
    try:
        # 前向传播
        y = model(x)
        
        print("\n✅✅✅ 模型运行成功！恭喜！✅✅✅")
        
        # 检查输出
        if isinstance(y, (list, tuple)):
            print(f"📦 输出类型: 列表 (深监督模式), 长度: {len(y)}")
            print(f"👉 最终层输出尺寸: {y[0].shape}")
        else:
            print(f"👉 输出尺寸: {y.shape}")
            
        print("\n这意味着你的 SDG-Block 已经成功缝合，且没有显存/维度报错。")
        print("可以直接开始 nnU-Net 训练了！")
            
    except Exception as e:
        print(f"\n❌ 运行报错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()