from beartype import BeartypeConf, beartype

# Accept ints where annotations specify floats
checker = beartype(conf=BeartypeConf(is_pep484_tower=True))
