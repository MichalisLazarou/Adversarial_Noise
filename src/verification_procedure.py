import torch
import torchvision
import numpy as np
from main import AdversarialAttack
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_attack = AdversarialAttack(device=device)
    imagenet_data = torchvision.datasets.ImageFolder('/home/michalislazarou/PhD/filelists/miniImagenet/ILSVRC2015/Data/CLS-LOC/train/', transform= adv_attack.weights.transforms())
    dataset_subset = torch.utils.data.Subset(imagenet_data, np.random.choice(len(imagenet_data), 500, replace=False))
    test_loader = torch.utils.data.DataLoader(dataset_subset, batch_size=1, shuffle=True, num_workers=8)

    ys = []
    y_preds = []
    for i, (x, y) in enumerate(test_loader):
        print(i)
        #setting up the new desired labels
        new_y = (y + 25) % 1000
        adv_images, _, predicted_cls = adv_attack.attack_training(x, new_y)
        ys.append(new_y.cpu().detach().numpy())
        print(new_y.shape, predicted_cls.shape)
        y_preds.append(predicted_cls.cpu().detach().numpy())
        print('Adv labels:', new_y)
        print('Adv preds:', predicted_cls)
    print('Adversarial accuracy: {}%'.format(accuracy_score(np.concatenate(ys, axis=0), np.concatenate(y_preds, axis=0))*100))
