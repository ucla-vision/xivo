import numpy as np

INDENT = 2
SPACE = " "
NEWLINE = "\n"
def to_json(o, level=0):
    """A custom json serializer that doesn't print every element of an array on its own
    new line. Modified slightly from
    https://stackoverflow.com/questions/10097477/python-json-array-newlines
    """
    ret = ""
    if isinstance(o, dict):
        ret += SPACE*INDENT*level + "{" + NEWLINE
        comma = ""
        for k,v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level+1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)
        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        if level==0:
            between = ",\n"
            ret += "[\n" + between.join([to_json(e, level+1) for e in o]) + "\n]"
        else:
            ret += "[" + ",".join([to_json(e, level+1) for e in o]) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += str(o)
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret
