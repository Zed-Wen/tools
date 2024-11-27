import os

# 你想要检查的路径
create_dir = '/path/to/your/directory'

# 检查路径是否存在
if not os.path.exists(create_dir):
    # 如果路径不存在，则创建它
    os.makedirs(create_dir)