import matplotlib.pyplot as plt

# 初始化三种loss的列表
plcc_losses = []
srcc_losses = []
total_losses = []

# 读取loss数据
with open("draw.txt", "r",) as f:
    for line in f:
        if "train/plcc_loss" in line:
            plcc_loss = float(line.split()[-1])
            plcc_losses.append(plcc_loss)
        elif "train/srcc_loss" in line:
            srcc_loss = float(line.split()[-1])
            srcc_losses.append(srcc_loss)
        elif "train/total_loss" in line:
            total_loss = float(line.split()[-1])
            total_losses.append(total_loss)

# 绘制loss图
# plt.plot(plcc_losses, label='PLCC Loss')
# plt.plot(srcc_losses, label='SRCC Loss')
plt.plot(total_losses, label='Total Loss')

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Losses over Time")
plt.legend()
plt.show()
