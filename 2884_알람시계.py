h , m = map(int,input().split)


if m >= 45:
    new_h = h
    new_m = m-45
    print(new_h, new_m)

elif m < 45 and h>0 :
    new_h = h-1
    new_m = m +15
    print(new_h, new_m)

else:
    new_h = h
    new_m = m +15
    print(new_h, new_m)
