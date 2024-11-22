function Contains(self, value)
    for _, tVal in ipairs(self) do
        if value == tVal then
            return true
        end
    end
    return false
end


function CharAt(str, pos)
    return string.sub(str, pos, pos)
end