import argparse
from collections import defaultdict
from itertools import cycle

from matplotlib import pyplot as plt; from matplotlib.lines import Line2D


def main(args):
    with open(args.loss_file, "r") as f:
        losses = [ loss_line.rstrip() for loss_line in f ]

    training_epochs, training_losses, training_losses_D = [], [], defaultdict(list)
    validation_epochs, validation_losses = [], defaultdict(list)

    if args.iters_per_epoch == -1:
        iters_per_epoch = int(losses[0].split(": ")[1])
        losses = losses[1:]
    else:
        iters_per_epoch = args.iters_per_epoch

    for i_line, line in enumerate(losses):
        if i_line + (3 if args.legacy else 5) >= len(losses):
            break

        if line.startswith("Epoch: "):
            epoch = int(line.split("Epoch: ")[1].split(",")[0])
            iter = int(line.split("Iter: ")[1].split(",")[0])
            loss_line = losses[i_line + 1]
            tot_loss = float(loss_line.split("total=")[1].split(" ")[0])
            training_epochs.append(epoch + iter / iters_per_epoch)
            training_losses.append(tot_loss)
            if "D_tot" in loss_line:
                training_losses_D["D_tot"].append(
                    float(loss_line.split("D_tot=")[1].split(" ")[0])
                )
                # training_losses_D["D_real"].append(
                #     float(loss_line.split("D_real=")[1].split(" ")[0])
                # )
                # training_losses_D["D_fake"].append(
                #     float(loss_line.split("D_fake=")[1].split(" ")[0])
                # )
                training_losses_D["G_GAN"].append(
                    float(loss_line.split("G_GAN=")[1].split(" ")[1][1:-1])
                )

        if line == "== Validation Loop ==":
            loss_line = losses[i_line + 2]
            tot_loss = float(loss_line.split("total=")[1].split(" ")[0])
            epoch_line = losses[i_line + (3 if args.legacy else 5)]
            epoch = int(epoch_line.split("Epoch ")[1].split(" ")[0])
            validation_epochs.append(epoch)
            validation_losses["total"].append(tot_loss)
            loss_line_comps = loss_line.split("total=")[1].split(" ")[1:]
            while loss_line_comps:
                el = loss_line_comps.pop()
                if el[0] == "(" and el[-1] == ")":
                    weighted_loss = float(el[1:-1])
                    loss_name = loss_line_comps.pop().split("=")[0]
                    # I dont think we care about this on validation dataset, only for training
                    if loss_name == "G_GAN":
                        continue
                    validation_losses[loss_name].append(weighted_loss)

    fig, ax = plt.subplots()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    ax.plot(training_epochs, training_losses, label="train - total", c=next(colors))
    validation_losses_total = validation_losses.pop("total")
    ax.plot(validation_epochs, validation_losses_total, label="valid - total", c=next(colors))
    for loss_name, losses in validation_losses.items():
        ax.plot(
            validation_epochs, losses, "--",
            label="valid - {} (weighted)".format(loss_name), c=next(colors)
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 1.2 * max(validation_losses_total))

    if training_losses_D:
        losses = training_losses_D.pop("G_GAN")
        ax.plot(training_epochs[-len(losses):], losses, "-.", label="G_GAN", c=next(colors))

        ax2 = ax.twinx()
        for loss_name, losses in training_losses_D.items():
            ax2.plot(training_epochs[-len(losses):], losses, "-.", label=loss_name, c=next(colors))
        ax2.set_ylabel("D loss")
        ax2.set_ylim(0, 1.2 * max(max(training_losses_D.values(), key=lambda losses: max(losses))))

        ax_ylims = ax.axes.get_ylim()
        ax_yratio = ax_ylims[0] / ax_ylims[1]
        ax2_ylims = ax2.axes.get_ylim()
        ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
        if ax_yratio < ax2_yratio:
            ax2.set_ylim(bottom=ax2_ylims[1] * ax_yratio)
        else:
            ax.set_ylim(bottom=ax_ylims[1] * ax2_yratio)

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
        new_handles = [Line2D([], [], c=h.get_color()) for h in handles]
        plt.legend(handles=new_handles, labels=labels, ncol=2)
    else:
        plt.legend(ncol=2)

    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("loss_file", type=str)

    parser.add_argument("--iters_per_epoch", type=int, default=-1)
    parser.add_argument("--legacy", action="store_true")

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())

