# -*- coding: utf-8 -*-

"""
    This script wraps the print methods [print, input], adding time stamps and identifiers for success and failed, like this:
    [+]2018-01-24 09:23:02 Engine start...
    [-]2018-01-24 09:23:21 Error! The argument timestep should not more than duration! Engine strike.

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2018/2/1
    $Annotation: Add sinput() function.
    $Author: xiyan19
"""


import time


def sprint(content):
    return print('[+]{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), content))


def fprint(content):
    return print('[-]{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), content))


def sinput(content):
    return input('[+]{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), content))


if __name__ == '__main__':
    print('- Don\'t execute me alone! You stupid!')
