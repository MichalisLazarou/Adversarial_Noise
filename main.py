import torchvision.utils
from torchvision.io import read_image, ImageReadMode
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from src.arguments import parse_option
import torch
import torchattacks

class AdversarialAttack:
    def __init__(self, device, attack = 'PGD'):
        # self.target_class = target_class
        self.device = device
        self.weights = EfficientNet_V2_M_Weights.DEFAULT
        self.model  = efficientnet_v2_m(weights=self.weights).eval().to(self.device)
        self.preprocess = self.weights.transforms()
        if attack=='PGD':
            self.atk = torchattacks.PGD(self.model, eps=1 / 255, alpha=1 / 225, steps=10, random_start=True)
        elif attack=='EAD':
            self.atk = torchattacks.EADL1(self.model, kappa=0, lr=0.01, max_iterations=100)
        else:
            print('Attack is not supported, initializing default attack PGD')
            self.atk = torchattacks.PGD(self.model, eps=1 / 255, alpha=1 / 225, steps=10, random_start=True)

        self.atk.set_mode_targeted_by_label()
        mean = torch.tensor(self.preprocess.mean, dtype=torch.float32)
        std = torch.tensor(self.preprocess.std, dtype=torch.float32)
        self.unnormalize = torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    def save_img(self, img, name):
        torchvision.utils.save_image(self.unnormalize(img).squeeze().cpu().detach(), '{}.png'.format(name))

    def attack_img(self, img_path, target_class):
        target_id = self.weights.meta['categories'].index(target_class)
        target_id = torch.Tensor([target_id]).long()
        img = read_image(img_path, mode=ImageReadMode.RGB)
        img = self.preprocess(img).unsqueeze(0)
        # self.atk.set_mode_targeted_by_label()
        img = img.to(self.device)
        while (True):
            # model, batch = model.to(device), batch.to(device)
            img = self.atk(img, target_id.to(device))
            adv_pred = self.model(img.to(self.device)).argmax(1)
            if adv_pred.item() == target_id.item():
                # attack_not_complete = False
                break
        prediction = self.model(img).squeeze().softmax(0)
        class_id = prediction.argmax().item()
        prediction_category = self.weights.meta["categories"][class_id]
        return img, prediction_category

    def attack_training(self, x, target_y):
        x, target_y = x.to(self.device), target_y.to(self.device)
        while (True):
            # model, batch = model.to(device), batch.to(device)
            x = self.atk(x, target_y)
            adv_pred = self.model(x.to(self.device)).argmax(1)
            if adv_pred.item() == target_y.item():
                # attack_not_complete = False
                break
        prediction = self.model(x).softmax(1)
        class_id = prediction.argmax().item()
        prediction_category = self.weights.meta["categories"][class_id]
        return x, prediction_category, prediction.argmax(1)


if __name__ == '__main__':
    params = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    adv_attack = AdversarialAttack(device=device, attack=params.attack)
    img, predicted_class = adv_attack.attack_img(target_class=params.target_class, img_path=params.img_path)
    adv_attack.save_img(img, '{}_prediction'.format(predicted_class))
    print('Task Complete!')

