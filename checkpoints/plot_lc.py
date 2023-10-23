import argparse

from matplotlib import pyplot as plt


def main(args):
    with open(args.loss_file, "r") as f:
        losses = [ loss_line.rstrip() for loss_line in f ]

    training_epochs, training_losses = [], []
    validation_epochs, validation_losses = [], []

    if args.iters_per_epoch == -1:
        iters_per_epoch = int(losses[0].split(": ")[1])
        losses = losses[1:]
    else:
        iters_per_epoch = args.iters_per_epoch

    for i_line, line in enumerate(losses):
        if i_line + 3 >= len(losses):
            break

        if line.startswith("Epoch: "):
            epoch = int(line.split("Epoch: ")[1].split(",")[0])
            iter = int(line.split("Iter: ")[1].split(",")[0])
            loss_line = losses[i_line + 1]
            tot_loss = float(loss_line.split("total=")[1].split(" ")[0])
            training_epochs.append(epoch + iter / iters_per_epoch)
            training_losses.append(tot_loss)

        if line == "== Validation Loop ==":
            loss_line = losses[i_line + 2]
            tot_loss = float(loss_line.split("total=")[1])
            epoch_line = losses[i_line + 3]
            epoch = int(epoch_line.split("Epoch ")[1].split(" ")[0])
            validation_epochs.append(epoch)
            validation_losses.append(tot_loss)

    plt.plot(training_epochs, training_losses, label="train")
    plt.plot(validation_epochs, validation_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("total_loss")
    plt.ylim(0, 1.2 * max(validation_losses))
    plt.legend()
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("loss_file", type=str)

    parser.add_argument("--iters_per_epoch", type=int, default=-1)

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())

