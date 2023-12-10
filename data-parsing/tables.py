import camelot

tables = camelot.read_pdf('sample.pdf', pages="1-end")

for table in tables:
    print(table.df)
    print("\n\n")