def main():
    import argparse
    import logging
    import time

    import matplotlib.pyplot as plt

    import torch
    from torch.utils.data import DataLoader, random_split

    from torchvision.datasets import MNIST, FashionMNIST
    from torchvision.transforms import ToTensor

    from te941 import NN1, NN2, NN3, NN4, NN5


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', help='which dataset to train on', required=True)
    parser.add_argument('--model', help='which model to train', required=True)
    parser.add_argument('--model_path', help='path to save the trained model weights', required=True)
    parser.add_argument('--plot_path', help='path to save the loss/accuracy vs epoch plot', required=True)
    args = parser.parse_args()


    TRAIN_SPLIT = 5/6
    BATCH_SIZE = 64

    LEARNING_RATE = 1e-3
    MOMENTUM = 0.99

    EPOCHS = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    logging.info(f'loading the {args.dataset} dataset')
    Dataset = eval(args.dataset)

    train_data = Dataset(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    train_samples = int(len(train_data) * TRAIN_SPLIT)
    validation_samples = len(train_data) - train_samples

    (train_data, validation_data) = random_split(
        train_data,
        [train_samples, validation_samples],
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        dataset=validation_data,
        batch_size=BATCH_SIZE,
    )


    logging.info(f'initializing {args.model} model')
    Model = eval(args.model)
    model = Model().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': [],
    }


    logging.info('training model')
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_accuracy = 0.0

        model.train()
        for (X, y) in train_dataloader:
            (X, y) = (X.to(device), y.to(device))

            prediction = model(X)
            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X)
            train_accuracy += (prediction.argmax(1) == y).type(torch.float).sum().item()

        train_loss /= train_samples
        train_accuracy /= train_samples

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)


        validation_loss = 0.0
        validation_accuracy = 0.0

        with torch.no_grad():
            model.eval()
            for (X, y) in validation_dataloader:
                (X, y) = (X.to(device), y.to(device))

                prediction = model(X)
                loss = loss_fn(prediction, y)

                validation_loss += loss.item() * len(X)
                validation_accuracy += (prediction.argmax(1) == y).type(torch.float).sum().item()

        validation_loss /= validation_samples
        validation_accuracy /= validation_samples

        history['validation_loss'].append(validation_loss)
        history['validation_accuracy'].append(validation_accuracy)


        logging.info(
            f'EPOCH: {epoch+1}/{EPOCHS}\n' +
            f'    train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}\n' +
            f'    validation loss: {validation_loss:.4f}, validation accuracy: {validation_accuracy:.4f}'
        )

    end_time = time.perf_counter()
    logging.info(f'total time taken to train the model: {end_time - start_time:.2f}')


    torch.save(model.state_dict(), args.model_path)


    epochs = range(1, EPOCHS + 1)

    (fig, ax1) = plt.subplots()
    ax2 = ax1.twinx()

    (p1,) = ax1.plot(epochs, history['train_loss'], label='train loss', color='#1f77b4')
    (p2,) = ax2.plot(epochs, history['train_accuracy'], label='train accuracy', color='#ff7f0e')
    (p3,) = ax1.plot(epochs, history['validation_loss'], label='validation loss', color='#2ca02c')
    (p4,) = ax2.plot(epochs, history['validation_accuracy'], label='validation accuracy', color='#d62728')

    ax1.legend(loc='center right', handles=[p1, p2, p3, p4])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')

    fig.tight_layout()
    fig.savefig(args.plot_path)


if __name__ == '__main__': main()
