import numpy as np

# predicts the price of US ETF stocks

def relu(x):
    return np.maximum(0, x)

def relu_diff(x):
    return (x > 0).astype(float)

class Model:
    def __init__(self, l1, l2, l3, l4, lr=0.01):
        self.LAYER1 = l1
        self.LAYER2 = l2
        self.LAYER3 = l3
        self.LAYER4 = l4
        self.lr = lr # learning rate

        self.a1 = np.zeros(self.LAYER1)
        self.a2 = np.zeros(self.LAYER2)
        self.a3 = np.zeros(self.LAYER3)
        self.a4 = np.zeros(self.LAYER4)

        self.w1 = np.random.randn(self.LAYER2, self.LAYER1) * (2 / np.sqrt(self.LAYER1))
        self.w2 = np.random.randn(self.LAYER3, self.LAYER2) * (2 / np.sqrt(self.LAYER2))
        self.w3 = np.random.randn(self.LAYER4, self.LAYER3) * (1 / np.sqrt(self.LAYER3))

        self.b1 = np.zeros(self.LAYER2)
        self.b2 = np.zeros(self.LAYER3)
        self.b3 = np.zeros(self.LAYER4)

        self.u1 = np.zeros(self.LAYER2)
        self.u2 = np.zeros(self.LAYER3)

    def forward(self):
        self.u1 = self.w1 @ self.a1 + self.b1
        self.a2 = relu(self.u1)

        self.u2 = self.w2 @ self.a2 + self.b2
        self.a3 = relu(self.u2)

        self.a4 = self.w3 @ self.a3 + self.b3
        return float(self.a4)

    def train(self, correct):
        diff = (self.a4 - correct)
        dw3 = np.outer(diff, self.a3)
        db3 = diff

        delta2 = (diff @ self.w3) * relu_diff(self.u2)
        dw2 = np.outer(delta2, self.a2)
        db2 = delta2

        delta1 = (delta2 @ self.w2) * relu_diff(self.u1)
        dw1 = np.outer(delta1, self.a1)
        db1 = delta1

        self.w1 = self.w1 - self.lr * dw1
        self.w2 = self.w2 - self.lr * dw2
        self.w3 = self.w3 - self.lr * dw3
        self.b1 = self.b1 - self.lr * db1
        self.b2 = self.b2 - self.lr * db2
        self.b3 = self.b3 - self.lr * db3

    def save(self):
        with open("train_data.txt", "w") as f:
            f.write("1\n")

            for i in range(self.LAYER2):
                for j in range(self.LAYER1):
                    f.write(f"{self.w1[i][j]} ")
                f.write("\n")

            for i in range(self.LAYER3):
                for j in range(self.LAYER2):
                    f.write(f"{self.w2[i][j]} ")
                f.write("\n")

            for i in range(self.LAYER4):
                for j in range(self.LAYER3):
                    f.write(f"{self.w3[i][j]} ")
                f.write("\n")

            for i in range(self.LAYER2):
                f.write(f"{self.b1[i]} ")
            f.write("\n")

            for i in range(self.LAYER3):
                f.write(f"{self.b2[i]} ")
            f.write("\n")

            for i in range(self.LAYER4):
                f.write(f"{self.b3[i]} ")
            f.write("\n")

    def load(self):
        with open("train_data.txt", "r") as f:
            if f.readline().strip() != "1":
                return 0

            for i in range(self.LAYER2):
                line = list(map(float, f.readline().split()))
                for j in range(self.LAYER1):
                    self.w1[i][j] = line[j]

            for i in range(self.LAYER3):
                line = list(map(float, f.readline().split()))
                for j in range(self.LAYER2):
                    self.w2[i][j] = line[j]

            for i in range(self.LAYER4):
                line = list(map(float, f.readline().split()))
                for j in range(self.LAYER3):
                    self.w3[i][j] = line[j]

            line = list(map(float, f.readline().split()))
            for i in range(self.LAYER2):
                self.b1[i] = line[i]

            line = list(map(float, f.readline().split()))
            for i in range(self.LAYER3):
                self.b2[i] = line[i]

            line = list(map(float, f.readline().split()))
            for i in range(self.LAYER4):
                self.b3[i] = line[i]

        return 1










