import os
import argparse

class MakePatterns():
    def __init__(self):
      self.cls_name = []
      self.patterns = []

    def Process(self,class_path,pattern_path):
        """ read class names """
        fh = open(class_path, 'r')
        class_names = [c.rstrip() for c in fh.readlines()]
        fh.close()

        """read pattern sentences"""
        gh = open(pattern_path,'r')
        line = [m.rstrip() for m in gh.readlines()]
        gh.close()

        self.cls_name = [l.split("\n")[0] for l in class_names]
        # add class names to texts
        for cls in self.cls_name:
            self.patterns.append(cls)

        self.sentenc = [k.split("\n")[0] for k in line]
        for stn in self.sentenc:
            for cls in self.cls_name:   
                stn_1= stn.replace("Object",cls,1)
                for cls in self.cls_name:
                    stn_new=stn_1.replace("Object",cls)
                    if stn_new not in (self.patterns):
                        self.patterns.append(stn_new)
        return self.patterns


def main(args):
    model = MakePatterns()
    result = model.Process(class_path=args.class_path, pattern_path=args.pattern_path)
    txt_embedding_path = args.txt_embedding_path
    with open(txt_embedding_path, 'w') as f:
        for line in result:
            f.write(line)
            f.write('\n')
        f.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--class_path', type=str, default='./data/ModelNet/modelnet40_shape_names.txt')
  parser.add_argument('--pattern_path', type=str, default='./data/patterns.txt')
  parser.add_argument('--txt_embedding_path', type=str, default='./data/ModelNet/bert_txt_embedding_modelnet40.txt')

  args = parser.parse_args()
  main(args)