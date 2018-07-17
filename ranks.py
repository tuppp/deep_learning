def add(values,weights):
    x=0
    y=0
    for i in range (0, len(values)):
        x=x+values[i]*weights[i]
        y=y+weights[i]
    return x/y