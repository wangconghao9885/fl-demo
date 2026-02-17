from torchvision import datasets, transforms

def main() -> None:
    transform = transforms.Compose([transforms.ToTensor()])
    datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    print("MNIST descargado en ./data")

if __name__ == "__main__":
    main()
