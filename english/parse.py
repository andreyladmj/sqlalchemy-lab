from rx import Observable


def prepare_text(row):
    if '<td class="ind gray">' in row:
        return row.replace('<td class="ind gray">', '').replace('</td>', '')

    if '<td><b>' in row and '</b></td>' in row:
        return row.replace('<td><b>', '').replace('</b></td>', '')

    if '<td class="tran">' in row:
        return row.replace('<td class="tran">', '').replace('</td>', '')

    return ''


def to_list(items):
    for i in range(0, len(items), 3):
        try:
            yield items[i], items[i + 1], items[i + 2]
        except IndexError:
            pass


def parse(filename):
    file = open(filename, encoding="utf8")

    return Observable.from_(file) \
        .map(prepare_text) \
        .map(lambda l: l.strip()) \
        .filter(lambda l: l != '') \
        .to_list() \
        .flat_map(to_list)


if __name__ == '__main__':
    ob = parse('words.html')
    ob.subscribe(lambda s: print(s))