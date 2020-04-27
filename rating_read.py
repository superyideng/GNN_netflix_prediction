import os
import time, datetime
PATH='training_set'
path_list = sorted(os.listdir(PATH))
with open('ratings.dat','w') as f:
    for i in path_list:
        with open(os.path.join(PATH,i),'r') as r:
            ll = r.readlines()
            movie_id = int(ll[0].strip()[:-1])
            for i in ll[1:]:
                line = i.strip().split(',')
                if len(line)==3:
                    user_id = str(int(line[0]))
                    rating = line[1]
                    tss = line[-1]
                    timeArray = time.strptime(tss, "%Y-%m-%d")
                    timeStamp = str(int(time.mktime(timeArray)))
                    if user_id and rating and tss and 1<=int(rating)<=5:
                        list_tmp = [str(user_id)]+[str(movie_id)]+[rating]+[timeStamp]
                        f.write('::'.join(list_tmp)+'\n')
