#!/usr/bin/env python

"""
Parse training log
"""

import os
import re
import argparse

import numpy as np

import inspect
import random
import sys

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt, mpld3
import matplotlib.legend as lgd
import matplotlib.markers as mks



fig = plt.figure()
###https://pythex.org/
def parse_log(path):
    regex_iteration = re.compile('(\d+):')
    regex_loss = re.compile('([\.\deE+-]+) avg')
    regex_iou = re.compile('Avg IOU: ([\.\deE+-]+)')
    regex_cat = re.compile('Class: ([\.\deE+-]+)')
    regex_obj = re.compile(', Obj: ([\.\deE+-]+)')
    regex_noobj = re.compile('No Obj: ([\.\deE+-]+)')
    regex_recall = re.compile('Recall: ([\.\deE+-]+)')

    iteration = 0
    loss = 0

    iterations = []
    losses = []

    iou = []
    cat = []
    obj = []
    noobj = []
    recall = []

    ious = []
    cats = []
    objs = []
    noobjs = []
    recalls = []

    batches = 0

    with open(path) as f:
        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
                iterations.append(iteration)
            else:
                iou_match = regex_iou.search(line)
                if iou_match:
                    batches = batches + 1
                    iou.append(float(iou_match.group(1)))

                cat_match = regex_cat.search(line)
                if cat_match:
                    cat.append(float(cat_match.group(1)))

                obj_match = regex_obj.search(line)
                if obj_match:
                    obj.append(float(obj_match.group(1)))

                noobj_match = regex_noobj.search(line)
                if noobj_match:
                    noobj.append(float(noobj_match.group(1)))

                recall_match = regex_recall.search(line)
                if recall_match:
                    recall.append(float(recall_match.group(1)))
                
            loss_match = regex_loss.search(line)
            if loss_match:
                loss = float(loss_match.group(1))
                losses.append(loss)
                if batches > 0:
                    ious.append(sum(iou)/batches)
                    cats.append(sum(cat)/batches)
                    objs.append(sum(obj)/batches)
                    noobjs.append(sum(noobj)/batches)
                    recalls.append(sum(recall)/batches)
                    batches = 0
                    del iou[:]
                    del cat[:]
                    del obj[:]
                    del noobj[:]
                    del recall[:]
                    iou = []
                    cat = []
                    obj = []
                    noobj = []
                    recall = []

    return (iterations, losses, ious, cats, objs, noobjs, recalls)

def plot_chart(data, trend):
    if trend == '0':
	    color = [0, 0, 0]

	    plt.clf()
	    # loss
	    plt.subplot(4, 2, 1)
	    plt.plot(data[0], data[1], label = '02', color = color,
		             linewidth = .8)
	    plt.title('loss')
	    plt.xlabel('iterations')
	    plt.ylabel('loss')
	    # iou
	    plt.subplot(4, 2, 2)
	    plt.plot(data[0], data[2], label = '02', color = color,
		             linewidth = .8)
	    plt.title('iou')
	    plt.xlabel('iterations')
	    plt.ylabel('iou')
	    # cat
	    plt.subplot(4, 2, 3)
	    plt.plot(data[0], data[3], label = '02', color = color,
		             linewidth = .8)
	    plt.title('cat')
	    plt.xlabel('iterations')
	    plt.ylabel('cat')
	    # obj
	    plt.subplot(4, 2, 4)
	    plt.plot(data[0], data[4], label = '02', color = color,
		             linewidth = .8)
	    plt.title('obj')
	    plt.xlabel('iterations')
	    plt.ylabel('obj')
	    # noobj
	    plt.subplot(4, 2, 5)
	    plt.plot(data[0], data[5], label = '02', color = color,
		             linewidth = .8)
	    plt.title('noobj')
	    plt.xlabel('iterations')
	    plt.ylabel('noobj')
	    # recall
	    plt.subplot(4, 2, 6)
	    plt.plot(data[0], data[5], label = '02', color = color,
		             linewidth = .8)
	    plt.title('recall')
	    plt.xlabel('iterations')
	    plt.ylabel('recall')

            plt.subplots_adjust(left=0.05, bottom=0, right=0.99, top=0.95, wspace=0.1, hspace=0.35)
	    #plt.savefig('./fig/fig')
    else:
	    color = [0, 0, 0]#[random.random(), random.random(), random.random()]

	    #plt.clf()
	    # loss
            fig.clf()
            ax = fig.add_subplot(421)
	    ax.plot(data[0], data[1], label = '02', color = [0, 0, 0],
		             linewidth = .8)

	    z = np.polyfit(data[0], data[1], 8)
            p = np.poly1d(z)
            ax.plot(data[0],p(data[0]),"r-", label='loss')

	    ax.set_title('loss')
	    ax.set_xlabel('iterations')
	    ax.set_ylabel('loss')
            ax.grid(True)

	    # iou
            ax = fig.add_subplot(422)
	    ax.plot(data[0], data[2], label = '02', color = [0, 0, 0],
		             linewidth = .8)

	    z = np.polyfit(data[0], data[2], 8)
            p = np.poly1d(z)
            ax.plot(data[0],p(data[0]),"r-", label='iou')

	    ax.set_title('iou')
	    ax.set_xlabel('iterations')
	    ax.set_ylabel('iou')
            ax.grid(True)

	    # cat
            ax = fig.add_subplot(423)
	    ax.plot(data[0], data[3], label = '02', color = [0, 0, 0],
		             linewidth = .8)

	    z = np.polyfit(data[0], data[3], 8)
            p = np.poly1d(z)
            ax.plot(data[0],p(data[0]),"r-", label='cat')

	    ax.set_title('cat')
	    ax.set_xlabel('iterations')
	    ax.set_ylabel('cat')
            ax.grid(True)

	    # obj
            ax = fig.add_subplot(424)
	    ax.plot(data[0], data[4], label = '02', color = [0, 0, 0],
		             linewidth = .8)

	    z = np.polyfit(data[0], data[4], 8)
            p = np.poly1d(z)
            ax.plot(data[0],p(data[0]),"r-", label='obj')


	    ax.set_title('obj')
	    ax.set_xlabel('iterations')
	    ax.set_ylabel('obj')
            ax.grid(True)

	    # noobj
            ax = fig.add_subplot(425)
	    ax.plot(data[0], data[5], label = '02', color = [0, 0, 0],
		             linewidth = .8)

	    z = np.polyfit(data[0], data[5], 8)
            p = np.poly1d(z)
            ax.plot(data[0],p(data[0]),"r-", label='noobj')

	    ax.set_title('noobj')
	    ax.set_xlabel('iterations')
	    ax.set_ylabel('noobj')
            ax.grid(True)

	    # recall
            ax = fig.add_subplot(426)
	    ax.plot(data[0], data[6], label = '02', color = [0, 0, 0],
		             linewidth = .8)

	    z = np.polyfit(data[0], data[6], 8)
            p = np.poly1d(z)
            ax.plot(data[0],p(data[0]),"r-", label='recall')

	    ax.set_title('recall')
	    ax.set_xlabel('iterations')
	    ax.set_ylabel('recall')
            ax.grid(True)

            fig.set_size_inches(15, 10, forward=True)
            fig.subplots_adjust(left=0.05, bottom=0, right=0.99, top=0.95, wspace=0.1, hspace=0.5)
	    #plt.savefig('./fig/fig.png')
            #mpld3.fig_to_html(fig, template_type="notebook")
            #mpld3.show()

def parse_args():
    description = ('Parse a darknet training log into graphs '
                   'containing loss')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--logfile',
                        help='Path to log file')


    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))

    parser.add_argument('--trend',
                        default='0',
                        help=('Calc trend for graphs '
                              '(default: \'%(default)s\')'))

    parser.add_argument('--interval',
                        default='60',
                        help=('Interval between updates '
                              '(default: \'%(default)s\')'))

    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    plt.ion()

    iteration = 0
    while True:
        if os.path.exists(args.logfile): 
            try:
                train_log = parse_log(args.logfile)
                plot_chart(train_log, args.trend)
            except:
                print "Parse log error"
	        plt.pause(int(args.interval))
                continue
        else:
            print "Log doesn't exists"
	    plt.pause(60)
            continue

 
        print str(iteration)
        iteration = iteration + 1
        #plt.show()
        plt.pause(int(args.interval))

if __name__ == '__main__':
    main()

