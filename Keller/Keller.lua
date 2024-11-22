
require("Keller.Stack")

local stack = Stack()

Push(stack, 1)

print(ToString(Pop(stack))) -- 1

Push(stack, 2)

print(ToString(Pop(stack))) -- 2

Push(stack, 3)

print(Peek(stack), ToString(stack)) -- 3 3

Push(stack, 4)

print(ToString(stack)) -- 3 4

Swap(stack)

print(ToString(stack)) -- 4 3

print(ToString(Duplicate(stack))) -- 4 3 3

print(ToString(Pop(stack))) -- 4 3

print(ToString(Clear(stack))) -- Empty

print(IsEmpty(stack)) -- true