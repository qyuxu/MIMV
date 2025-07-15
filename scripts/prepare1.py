if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str)
    parser.add_argument('--file2_1', type=str)
    parser.add_argument('--file2_2', type=str)
    parser.add_argument('--out1', type=str)
    parser.add_argument('--out2', type=str)
    args = parser.parse_args()

    main(args.file1, args.file2_1, args.file2_2, args.out1, args.out2)

