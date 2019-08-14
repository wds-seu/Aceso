#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
import visdom
import time
import numpy as np
import logging
from config import DefaultConfig

opt = DefaultConfig()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')


class Visualizer(object):
    """
    封装了visdom的基本操作
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 保存('loss', 23) 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        :param env: 环境名称
        :param kwargs: 参数
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次绘制多个
        :param d: dict (name, value)
        """
        for k, v in d.items():
            self.plot(k, v)

    def plot(self, win_name, y, **kwargs):
        x = self.index.get(win_name, 0)
        # visdom接收numpy或者tensor数据，所以需要进转换
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=win_name,
                      opts=dict(title=win_name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[win_name] = x + 1

    def plot_multi(self, name, y, **kwargs):
        """ 一个图中绘制多个曲线 """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      **kwargs
                      )
        self.index[name] = x + 1

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_img', t.Tensor(3, 64, 64))
        self.img('input_img', t.Tensor(100, 164, 64))
        self.img('input_img', t.Tensor(100, 3, 64, 64), nrows=10)
        !!! don't
        --self.img('input_img', t.Tensor(100, 64, 64), nrows=10)--
        !!!
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        :param info: 日志信息
        :param win: 日志panel
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info
        ))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        self.function 等价于 self.vis.function
        自定义的plot、log、plot_many等除外
        """
        return getattr(self.vis, name)

    def calculate_and_show(self, cm_value, together_calculate=True):
        if together_calculate:
            accuracy = 100 * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2] + cm_value[3][3]) / (
                cm_value.sum())
            precision = 100 * (cm_value[0][0] / cm_value[:, 0].sum() + cm_value[1][1] / cm_value[:, 1].sum() +
                               cm_value[2][2] / cm_value[:, 2].sum() + cm_value[3][3] / cm_value[:,
                                                                                        3].sum()) / 4
            recall = 100 * (cm_value[0][0] / cm_value[0, :].sum() + cm_value[1][1] / cm_value[1, :].sum() +
                            cm_value[2][2] / cm_value[2, :].sum() + cm_value[3][3] / cm_value[3, :].sum()) / 4
            f1 = 2 * precision * recall / (precision + recall)
            return accuracy, precision, recall, f1
        else:
            precision_P = 100 * cm_value[0][0] / cm_value[:, 0].sum()
            recall_P = 100 * cm_value[0][0] / cm_value[0, :].sum()
            f1_P = 2 * precision_P * recall_P / (precision_P + recall_P)

            precision_I = 100 * cm_value[1][1] / cm_value[:, 1].sum()
            recall_I = 100 * cm_value[1][1] / cm_value[1, :].sum()
            f1_I = 2 * precision_I * recall_I / (precision_I + recall_I)

            precision_O = 100 * cm_value[2][2] / cm_value[:, 2].sum()
            recall_O = 100 * cm_value[2][2] / cm_value[2, :].sum()
            f1_O = 2 * precision_O * recall_O / (precision_O + recall_O)

            precision_N = 100 * cm_value[3][3] / cm_value[:, 3].sum()
            recall_N = 100 * cm_value[3][3] / cm_value[3, :].sum()
            f1_N = 2 * precision_N * recall_N / (precision_N + recall_N)
            return [precision_P, recall_P, f1_P], [precision_I, recall_I, f1_I], \
                   [precision_O, recall_O, f1_O], [precision_N, recall_N, f1_N]

    def plot_laprf(self, data, env):
        """
        show the loss、accuracy、precision、recall and f1 using visdom
        :param data:
        :return:
        """
        train_accuracy, train_precision, train_recall, train_f1 = data
        self.plot(env + '_accuracy', train_accuracy)
        self.plot(env + '_precision', train_precision)
        self.plot(env + "_recall", train_recall)
        self.plot(env + "_f1", train_f1)

    def plot_lprf_dependent(self, data, env):
        result_p, result_i, result_o, result_n = data
        self.plot(env + '_accuracy_p', result_p[0])
        self.plot(env + '_accuracy_i', result_i[0])
        self.plot(env + '_accuracy_o', result_o[0])
        self.plot(env + '_accuracy_n', result_n[0])

        self.plot(env + "_recall_p", result_p[1])
        self.plot(env + "_recall_i", result_i[1])
        self.plot(env + "_recall_o", result_o[1])
        self.plot(env + "_recall_n", result_n[1])

        self.plot(env + "_f1_p", result_p[2])
        self.plot(env + "_f1_i", result_i[2])
        self.plot(env + "_f1_o", result_o[2])
        self.plot(env + "_f1_n", result_n[2])
