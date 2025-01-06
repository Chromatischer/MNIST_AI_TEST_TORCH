---Stack implementation
---@class Stack
---@field data table the data in the stack
---@field current number the current index of the stack
---@return table Stack the stack
function Stack()
    return {
        data = {},
        current = 0
    }
end

function Push(self, element)
    table.insert(self.data, element)
    self.current = self.current + 1
end

function Pop(self)
    table.remove(self.data, self.current)
    self.current = self.current - 1
    return self
end

function PopReturn(self)
    local element = self.data[self.current]
    table.remove(self.data, self.current)
    self.current = self.current - 1
    return element
end

function Peek(self)
    return self.data[self.current]
end

function IsEmpty(self)
    return self.current == 0
end

function ToString(self)
    local str = ""
    for i = 1, #self.data do
        str = str .. self.data[i] .. " "
    end
    if IsEmpty(self) then
        str = "Empty"
    end
    return str
end

function Clear(self)
    self.data = {}
    self.current = 0
    return self
end

function Duplicate(self)
    table.insert(self.data, self.data[self.current])
    return self
end

function Swap(self)
    local temp = self.data[self.current]
    self.data[self.current] = self.data[self.current - 1]
    self.data[self.current - 1] = temp
    return self
end

function Rotate(self, n)
    local temp = {}
    for i = 1, n do
        table.insert(temp, self.data[self.current])
        table.remove(self.data, self.current)
        self.current = self.current - 1
    end
    for i = 1, n do
        table.insert(self.data, temp[n - i + 1])
        self.current = self.current + 1
    end
    return self
end