#!/usr/bin/env python  
# -*- coding:utf-8 _*-
import os
import xlrd
import codecs
import csv
import re


def load_data_xls(file_path):
    """
    从表格中读取训练数据
    :return:
    """
    # 路径前加 r，读取的文件路径
    if file_path == '':
        file_path = r'./data/PICO'

    P = []
    I = []
    O = []
    dirs = os.listdir(file_path)
    # 输出所有文件和文件夹
    for file in dirs:  # 文件夹下的子文件
        # print(file)
        if file.endswith('.xls'):
            childPath = os.path.join(file_path, file)
            # print(childPath)
            wb = xlrd.open_workbook(childPath)
            # 获取workbook中所有的表格
            sheets = wb.sheet_names()
            # print(sheets)
            for i in range(len(sheets)):
                if not sheets[i].isdigit():
                    continue
                sheet = wb.sheet_by_index(i)
                nrows = sheet.nrows
                for row in range(nrows):
                    label = sheet.cell(row, 0).value
                    sentence = sheet.cell(row, 1).value
                    if label == 'P':
                        P.append(sentence)
                    if label == 'I':
                        I.append(sentence)
                    if label == 'O':
                        O.append(sentence)
    print('训练集数量：')
    print('P    :', len(P))
    print('I    :', len(I))
    print('O    :', len(O))
    # print(P)
    save_to_csv(P, './data/P.csv')
    save_to_csv(I, './data/I.csv')
    save_to_csv(O, './data/O.csv')

    text_save(P, './data/P.txt')
    text_save(I, './data/I.txt')
    text_save(O, './data/O.txt')


def save_to_csv(datas, filePath):
    """
    将datas中的数据记录到filePath中去
    :param datas:  list类型的数据
    :param filePath:  csv的路径
    :return:
    """
    file_csv = codecs.open(filePath, 'w', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def text_save(data, filename):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'w', encoding='utf-8')
    for i in range(len(data)):
        # s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        # s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        s = re.sub(r'[^\x00-\x7f]', '', data[i])
        s = s + '\n'
        print(s)
        file.write(s)
    file.close()
    print("保存文件成功")


if __name__ == '__main__':
    load_data_xls('')