# coding=utf-8

# Created:    on June 7, 2018 20:10
# @Author:    xxoospring

import os
import time


def get_file_name(path, path_append=False):
    ret = []
    if not path.endswith('/'):
        path += '/'
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            # print(root) #current path
            # print(dirs) #sub directories in current path
            # print(files) #all files in this path, directories not included
            break
        if not path_append:
            return files
        else:
            for it in files:
                ret.append(path+it)
            return ret
    warm_print('can\'t find any files in this directory!')
    return None


def get_dir_name(path, path_apppend=False):
    ret = []
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            # print(root) #current path
            # print(dirs) #sub directories in current path
            # print(files) #all files in this path, directories not included
            pass
            if not path_apppend:
                return dirs
            else:
                for it in dirs:
                    ret.append(path+it+'/')
            return ret
    print('Path Not Exist')
    return None


def file_exist(path, file_name):
    if not os.path.exists(path):
        return False
    for _, __, files in os.walk(path):
        if files.__contains__(file_name):
            return True
    return False


# return all files end up with suffix in file_path
def file_filter(file_path, suffix, path_append=False):
    lst = get_file_name(file_path, path_append)
    _suffix = [suffix.upper(), suffix.lower(), suffix]
    if lst:
        return [item for item in lst if os.path.splitext(item)[-1][1:] in _suffix]

    else:
        return []


def warm_print(s):
    print('\033[1;35m%s\033[0m' % s)


# only support mixture of string, float, integer,
def enhance_join(lst, suffix=' '):
    ret = []
    for it in lst:
        if type(it) == str:
            ret.append(it)
        else:
            ret.append(str(it))
    return suffix.join(ret)


def list_to_file(file_name, lst, suffix=' '):
    assert os.path.exists(os.path.dirname(file_name)), "File Path Not Exist\n"
    assert type(lst) == list, "Param Type Error\n"
    with open(file_name, 'w', encoding='utf-8') as fw:
        for line in lst:
            if type(line) == tuple:
                w_string = enhance_join(list(line), suffix)
            elif type(line) == list:
                w_string = enhance_join(list(line), suffix)
            elif type(line) == str:
                w_string = line
            elif type(line) == float or type(line) == int:
                w_string = str(line)
            else:
                warm_print("Unsupport Type")
                raise ValueError
            fw.write(w_string+'\n')


def file_to_list(file_name, suffix=None):
    ret = []
    with open(file_name, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip()
            if suffix is not None:
                item = line.split(suffix)
                ret.append(item)
                continue
            ret.append(line)
    return ret


def get_cur_date():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())).replace(':', '_')


# def write_excel_xls(path, sheet_name, value):
#     index = len(value)  # 获取需要写入数据的行数
#     workbook = xlwt.Workbook()  # 新建一个工作簿
#     sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
#     for i in range(0, index):
#         for j in range(0, len(value[i])):
#             sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
#     workbook.save(path)  # 保存工作簿
#     print("xls格式表格写入数据成功！")

# add an "append_suffix" at "where" of the file names
def files_rename(directory, search_suffix='wav', append_suffix='NONE', where='pre'):
    files = file_filter(directory, search_suffix, False)
    assert len(files), 'No Files Founded!'
    if where == 'pre':
        for f in files:
            os.rename(os.path.join(directory, f), os.path.join(directory, append_suffix+f))
    else:
        for f in files:
            if not append_suffix.startswith('.'):
                append_suffix = '.' + append_suffix
            new_name = f[:-len(append_suffix)]+append_suffix[1:]+f[-len(append_suffix):]
            os.rename(os.path.join(directory, f), os.path.join(directory, new_name))
