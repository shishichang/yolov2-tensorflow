import os
import re


PATTERN = ('.jpg', '.jpeg')
def find_files(directory, pattern=PATTERN):
  files = []
  for path, d, filelist in os.walk(directory):
      for filename in filelist:
          if filename.lower().endswith(pattern):
              files.append(os.path.join(path, filename))
  return files

def map2classnames(labelmap_file):
    classes = []
    f = open(labelmap_file, 'r')
    pat = 'display_name'
    for line in f.readlines():
        if re.search(pat, line):
            line_strs = line.split('"')
            class_name = line_strs[-2]
            classes.append(class_name)
    f.close()
    return classes
