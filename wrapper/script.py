import pcf
import sys
import csv
import array


if len(sys.argv) == 4:

  with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    users = array.array('i', [])
    items = array.array('i', [])
    ratings = array.array('f', [])
    for row in csv_reader:
      if line_count != 0:
        users.append(int(row[0]))
        items.append(int(row[1]))
        ratings.append(float(row[2]))
      line_count +=1
  
  print(pcf.predict(users, items, ratings, int(sys.argv[2]), int(sys.argv[3])))
else:
  print("Usage: script.py filepath user item")
