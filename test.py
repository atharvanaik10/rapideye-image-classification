import dbf 

table = dbf.Table('data/CDL_2013_clip_20170525181724_1012622514.tif.vat.dbf')

print(table)

table.open(dbf.READ_WRITE)
count = 0
for record in table:
    print(record)
    print("------")
    count += 1
    if count == 10:
        break