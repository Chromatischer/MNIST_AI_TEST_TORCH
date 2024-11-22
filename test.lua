function iterativ(n)
    erg = 1
    for i = 2, n, 1 do
        erg = 2 * erg + 1
    end
    return erg
end

function unsers(n)
    res = 1
    for i = 1, n, 1 do
        res = 2 * res
    end
    return res - 1
end

print(iterativ(4))
print(unsers(4))