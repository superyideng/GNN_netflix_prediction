import os
import time, datetime
PATH='training_set'
path_list = sorted(os.listdir(PATH))
user_set=set()
with open('users.dat','w') as f:
    for i in path_list:
        with open(os.path.join(PATH,i),'r') as r:
            ll = r.readlines()
            for i in ll[1:]:
                line = i.strip().split(',')
                if len(line)==3:
                    user_id = str(int(line[0]))
                    if user_id:
                        user_set.add(user_id)
    user_set=sorted(list(user_set))
    for i in user_set:
        f.write(i+'\n')
