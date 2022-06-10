
class Map:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inputs):
        return [self.transform(x) for x in inputs]


class ApplyN:

    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, input):
        return [self.transform(input) for _ in range(self.n)]


class LeaveOneOut:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        outputs = []
        for i in range(len(self.transforms)):
            output = input
            for j in range(len(self.transforms)):
                if j != i:
                    output = self.transforms[j](output)
            outputs.append(output)
        return outputs


class AddOne:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        outputs = []
        for i in range(len(self.transforms)):
            output = self.transforms[i](input)
            outputs.append(output)
        return outputs
