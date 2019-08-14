
n = int(input('>>>'))


try:
    result = 1/n
    print(result)
except ZeroDivisionError:
    #print("da co loi: ", e)
    raise ZeroDivisionError
