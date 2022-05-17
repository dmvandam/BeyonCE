from enum import Enum

class GridPropertyUnit(Enum):
    """All the available units for disk properties."""
    
    ECLIPSE_DURATION = "$t_{ecl}$"
    DEGREE = "$^o$"
    NONE = "-"

    def __str__(self) -> str:
        """
        This method serves as the print value for this class

        Returns
        -------
        unit : str
            Human readable string representation of the enum value selected.
        """
        words = self.name.split("_")
        formatted_words = []
        
        for word in words:
            formatted_words.append(word.capitalize())
        
        unit = " ".join(formatted_words)
        
        return unit

    def get_unit(self) -> str:
        """
        This method is used to parse the unit in a pretty way.

        Returns
        -------
        unit : str
            LaTeX formatted version of enum name.
        """
        return self.value