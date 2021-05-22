import json

# Opening JSON file
f = open('/home/cyberdome/data/vg/relationships.json',)
  
# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list
for i in data['relationships']:
    print(i)
    break
  
# Closing file
f.close()