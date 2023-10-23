import torch


def to_one_hot(number, max_val, min_val=0):
    assert max_val > min_val, "illegal parameters, max_val is not larger than min_val"
    hot = torch.zeros(max_val - min_val)
    hot[number] = 1
    return hot


def from_one_hot(hot):
    index = torch.where(hot)
    assert len(index) == 1, "more than one value is 1 in one hot vector"

    return index[0]


if __name__ == "__main__":
    hot = to_one_hot(4, 0, 5)
    print(hot)
    val = from_one_hot(hot)
    print(val)
