# Tests

To run all the tests, start from the root and simply type

```bash
cd test/
pytest .
```

To run a specific test , type

```bash
cd test/
python test_<name1>.py Test<name2>.test_<name3>
```

where `<name1>, <name2>, <name3>` are the names of the test file, the class and the test function, respectively.

By default, tests are not run on GPU to enfore reproducibility.  
