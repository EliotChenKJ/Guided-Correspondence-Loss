import time
import torch
import utils
from generate import generate


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 参数设置
    size = (224, 224)
    use_layout = True
    for name in [4, 5, 6]:
        utils.create_folder(f"outputs/{name}")
        img_path = f"./data/reference/{name}.jpg"

        reference_img = utils.load_img(img_path).to(device)
        init_img = torch.randn((1, 3, *size), device=device)

        start = time.time()
        print(f"img: {img_path}")
        generate(init_img, reference_img, device, name, use_layout)
        end = time.time()
        print("Time for one image: ", (end - start) / 60.0)


if __name__ == '__main__':
    main()
    print("Done")
