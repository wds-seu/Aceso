#!/usr/bin/env python
# -*- coding:utf-8 _*-
import torch as t
import time
import logging
from config import DefaultConfig
opt = DefaultConfig()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要提供了save和load两个方法
    提供快速加载和保存模型的接口
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))   # 模型的默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        :param path: 路径
        :return: 模型
        """
        return self.load_state_dict(t.load(path))

    def save(self, name=None, epoch=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        :param name: 模型名字
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + "_epoch" + str(epoch) + '_'
            name = time.strftime(prefix + '%Y-%m-%d_%H-%M-%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(t.nn.Module):
    """
    reshape to (batch_size, dim_length)
    """
    def __init__(self):
        super(Flat, self).__init__()

    def foward(self, x):
        return x.view(x.size(0), -1)

