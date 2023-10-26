import torch

# X = torch.randn((6,5))
X = torch.tensor([
        [
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ]
        ],
        [
            [
                [9, 10],
                [11, 12]
            ],
            [
                [13, 14],
                [15, 16]
            ]
        ]
    ],dtype=float
)
Y = torch.tensor(
    [
        [
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20],
            [21,22,23,24]
        ]
    ]
    
)
print(X.shape, len(X.shape))

print(X.mean(dim=(0, 2, 3), keepdim=True))