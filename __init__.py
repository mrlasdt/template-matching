# # Define package-level variables
# __version__ = '0.0'

# Import modules
from .src.template_matching import TemplateMatcher
from .src.dto import PageAligned, DocumentAligned
# Expose package contents
__all__ = ["TemplateMatcher", "PageAligned", "DocumentAligned"]
