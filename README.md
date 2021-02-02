This add-on for Anki provides a window that shows you notes with similar content to the one you're currently editing. It is useful for ensuring that new notes you create don't have differently worded duplicates hiding in your deck.

To install the add-on, enter code `1919894971` into Anki's add-on manager. You'll also need to have the `sklearn` and `lxml` python packages installed. 

To use the add-on, choose "Show Similar Notes" from the "Tools" menu of the main window. As you type, the window will show similar notes, ranked by the dot product of their tf-idf vectors. Currently, this search is performed with exhaustive enumeration, which may be too slow for decks with millions of notes. 

Pull requests for more sophisticated document models like LDA, search indexes like a BK-tree, or other improvements are welcome. 

