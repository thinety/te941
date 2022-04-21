def main():
    import argparse
    import logging

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay

    import torch
    from torch.utils.data import DataLoader

    from torchvision.datasets import MNIST, FashionMNIST
    from torchvision.transforms import ToTensor

    from te941 import NN1, NN2, NN3, NN4, NN5


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', help='which dataset to test on', required=True)
    parser.add_argument('--model', help='which model to test', required=True)
    parser.add_argument('--model_path', help='path to the trained model weights', required=True)
    parser.add_argument('--confusion_matrix_path', help='path to save the confusion matrix plot', required=True)
    args = parser.parse_args()


    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    logging.info(f'loading the {args.dataset} dataset')
    Dataset = eval(args.dataset)

    test_data = Dataset(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
    )


    logging.info(f'initializing {args.model} model')
    Model = eval(args.model)
    model = Model().to(device)

    logging.info(f'loading model weights from `{args.model_path}`')
    model.load_state_dict(torch.load(args.model_path))


    logging.info('testing model')
    predictions = []

    with torch.no_grad():
        model.eval()
        for (X, _) in test_dataloader:
            X = X.to(device)

            prediction = model(X)

            predictions.extend(prediction.argmax(1).cpu().numpy())

    predictions = np.array(predictions)

    logging.info(
        '\n' + \
        classification_report(
            predictions,
            test_data.targets,
            target_names=test_data.classes,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        predictions,
        test_data.targets,
        display_labels=test_data.classes,
        xticks_rotation='vertical',
    )
    plt.tight_layout()
    plt.savefig(args.confusion_matrix_path)


if __name__ == '__main__': main()
