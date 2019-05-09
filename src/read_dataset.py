import os
import re
import numpy as np
import tqdm

re_empty = re.compile(r'^\s*$')
re_author = re.compile(r'^by (.*)')
re_attrs = re.compile(r'^([\w\s]+):\s?(.*)$')

def read_dataset(enc, path, combine):
    paths = []
    for (dirpath, _, fnames) in os.walk(path):
        for fname in fnames:
            paths.append(os.path.join(dirpath, fname))

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        with open(path, 'r') as fp:
            cons_empty_lines = 0
            title = None
            author = None
            attrs = {}
            has_attrs = False
            for index, line in enumerate(fp):
                if re_empty.match(line):
                    cons_empty_lines += 1
                    continue
                else:
                    if title is None:
                        title = line
                    elif author is None:
                        author_match = re_author.match(line, re.IGNORECASE)
                        if author_match:
                            author = author_match.group(1) or ''

                    if has_attrs and cons_empty_lines >= 3:
                        if 'Language' not in attrs or not re.match(r'english', attrs['Language'],
                                                                   flags=re.IGNORECASE):
                            break
                        lines = fp.readlines()[index:]
                        raw_text += ''.join(lines)
                        if len(raw_text) >= combine:
                            tokens = np.stack(enc.encode(raw_text))
                            token_chunks.append(tokens)
                            raw_text = ''
                        else:
                            raw_text += '<|endoftext|>'
                        break

                    attr_match = re_attrs.match(line)
                    if attr_match and attr_match.group(1) and attr_match.group(2):
                        has_attrs = True
                        attrs[attr_match.group(1)] = attr_match.group(2)

                    cons_empty_lines = 0
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks
