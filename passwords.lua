require("Utils")

--get user string input from console
print("Enter your password:")
--get user input
Entered = io.read("l")

Entered = string.upper(Entered)

Numbers = {"0","1","2","3","4","5","6","7","8","9"}
Letters = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"}
Special = {"-", "#", "."}

--check if the input contains illegal Letters
for i = 1, #Entered do
    local l = string.sub(Entered, i, i)
    if not Contains(Numbers, l) and not Contains(Letters, l) and not Contains(Special, l) then
        print("Illegal char: '" .. l .. "' at position: " .. i)
        os.exit(-1)
    end
end
print("Password Chars are legal!")

StartsWithLetter = false
EndsWithLetter = false

if Contains(Letters, string.sub(Entered, 1, 1)) then
    StartsWithLetter = true
end

if Contains(Letters, string.sub(Entered, #Entered, #Entered)) then
    EndsWithLetter = true
end

print("Password starts with letter: " .. (StartsWithLetter and "Yes" or "No"))
print("Password ends with letter: " .. (EndsWithLetter and "Yes" or "No"))

NumberOfSpecialCharacters = 0
NumberOfNumbers = 0

for i = 1, #Entered do
    local l = string.sub(Entered, i, i)
    if Contains(Special, l) then
        NumberOfSpecialCharacters = NumberOfSpecialCharacters + 1
    end
    if Contains(Numbers, l) then
        NumberOfNumbers = NumberOfNumbers + 1
    end
end

print("Number of special characters: " .. NumberOfSpecialCharacters)
print("Number of numbers: " .. NumberOfNumbers)

--Check that every special character is followed by a number, ignoring letters in between and vice versa

NumberSpecialPairwise = false

for i = 1, #Entered do
    local l = CharAt(Entered, i)
    if Contains(Special, l) then
        for j = i, #Entered do
            local l2 = CharAt(Entered, j)
            if Contains(Numbers, l2) then
                print("Special " .. l .. " character followed by number: Yes")
                NumberSpecialPairwise = true
                break
            end
        end
    end

    if Contains(Numbers, l) then
        for j = i, #Entered do
            local l2 = CharAt(Entered, j)
            if Contains(Special, l2) then
                print("Number " .. l .. " followed by special character: Yes")
                NumberSpecialPairwise = true
                break
            end
        end
    end
end

print("FollowRule: " .. (NumberSpecialPairwise and "Yes" or "No"))

if not (NumberOfSpecialCharacters == NumberOfNumbers) then
    print("Number of Special Characters and Numbers are not equal (" .. NumberOfSpecialCharacters .. " vs " .. NumberOfNumbers .. ")")
end

if not (StartsWithLetter and EndsWithLetter) then
    print("Password does not start and end with a letter")
end

if not NumberSpecialPairwise then
    print("Password does not follow the rule that every special character has to be followed by a number and vice versa")
end

if not (NumberOfSpecialCharacters == NumberOfNumbers) or not (StartsWithLetter and EndsWithLetter) or not NumberSpecialPairwise then
    print("Password is not VALID")
else
    print("Password is VALID")
end