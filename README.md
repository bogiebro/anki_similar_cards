This addon for Anki provides a window that shows you noes with similar content to the one you're currently editing. It is useful for ensuring that you don't already have notes for the same information in your deck when you're creating new ones. 

To install the addon, clone the repository to `~/.local/share/Anki2/addons21` and run `pip install`. 

To use the addon, choose "Show Similar Cards" from the "Tools" menu of the main window. As you type, the window will show similar notes, ranked by the dot product of their tf-idf vectors. Currently, this search is performed with exhaustive enumeration, which may be too slow for decks with millions of notes. 

Pull requests for more sophisticated document models like LDA, search indexes like a BK-tree, or other improvements are welcome. 