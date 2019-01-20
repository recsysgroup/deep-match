def builder(type):
    def print_type():
        print type

    return print_type


t1 = builder(1)
t2 = builder(2)
t3 = builder(3)

t1()
t2()
t3()
