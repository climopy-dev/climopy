# Copied from: https://github.com/matplotlib/matplotlib/blob/master/doc/sphinxext/custom_roles.py
# The :rc: role inputs relevant rcParam
from docutils import nodes
from os.path import sep


def rcparam_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    rendered = nodes.Text('rcParams["{}"]'.format(text))

    source = inliner.document.attributes['source'].replace(sep, '/')
    rel_source = source.split('/docs/', 1)[1]

    # Point to matplotlib documentation when :rc: role used
    # Alternatively point to my rcmod module?
    # levels = rel_source.count('/')
    # refuri = ('../' * levels +
    #           'tutorials/introductory/customizing.html#matplotlib-rcparams')
    refuri = 'https://matplotlib.org/tutorials/introductory/customizing.html#matplotlib-rcparams'

    ref = nodes.reference(rawtext, rendered, refuri=refuri)
    return [nodes.literal('', '', ref)], []


def setup(app):
    app.add_role("rc", rcparam_role)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
