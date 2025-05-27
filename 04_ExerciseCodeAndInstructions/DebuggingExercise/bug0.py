#bug 0, cause an exception to when run
def fReplaceText(strInput: str, strText: str, intStart: int, intStop: int) -> str:
    """
    Replace the text between intStart and intStop with strText
    """
    #copy the string
    strReturn = strInput[:]
    # update the section of the string with inputted text
    strReturn = strReturn[:intStart] + strText + strReturn[intStop:]
    return strReturn

def fFindAndReplaceInStr(strInput: str, strOld: str, strNew: str) -> str:
    """
    replace strOld with strNew in strInput
    """
    intLenOld = len(strOld)
    intStart = strInput.find(strOld)
    # Add return here
    return fReplaceText(strInput, strNew, intStart, intStart+intLenOld)


#fix famous misquote
print(fFindAndReplaceInStr("Luke, I am your father.", 'Luke', 'No'))
#Hint: run the file in debug mode, VSCode should catch the exception.
#Then try the Error line of code in DEBUG CONSOLE, check these strings, how to make it right?
