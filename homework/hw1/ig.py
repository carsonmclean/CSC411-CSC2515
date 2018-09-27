import math

# use this to double check my IG calculation using the values in course lecture slides Pg.24/33
# http://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec03-slides.pdf
real = 49
fake = 51
total = real + fake
print(real, fake, total)

print("Parent Entropy:")
parent_entropy = - (real/total*math.log2(real/total)) - (fake/total*math.log2(fake/total))
print(parent_entropy)

# print("Splitting on: " + str(xi))
# contains = Y[Y['titles'].str.contains(xi)]
con_real = 24
con_fake = 1
con_total = con_real + con_fake
print(con_real, con_fake)
# Andy Hayden -> https://stackoverflow.com/questions/17097643/search-for-does-not-contain-on-a-dataframe-in-pandas
# absent = Y[~Y['titles'].str.contains(xi)]
abs_real = 25
abs_fake = 50
abs_total = abs_real + abs_fake
print(abs_real, abs_fake)

if con_total > 0:
    con_entr = ((con_total/(con_total+abs_total)) * (- (con_real/con_total*math.log2(con_real/con_total)) - (con_fake/con_total*math.log2(con_fake/con_total))))
else:
    con_ent = 0
if abs_total > 0:
    abs_entr = ((abs_total/(con_total+abs_total)) * (- (abs_real/abs_total*math.log2(abs_real/abs_total)) - (abs_fake/abs_total*math.log2(abs_fake/abs_total))))
else:
    abs_entr = 0
ch_entropy = con_entr + abs_entr
print("Child Entropy: " + str(ch_entropy))

IG = parent_entropy - ch_entropy
print("Information Gain: " + str(IG))