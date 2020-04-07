/*
 * Anserini: A Lucene toolkit for replicable information retrieval research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.anserini.collection;

import edu.unh.cs.treccar_v2.Data;
import edu.unh.cs.treccar_v2.read_data.DeserializeData;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;

/**
 * A document collection for the TREC Complex Answer Retrieval (CAR) Track.
 * This class provides a wrapper around <a href="https://github.com/TREMA-UNH/trec-car-tools">tools</a>
 * provided by the track for reading the <code>cbor</code> format.
 * Since a collection is assumed to be in a directory, place the <code>cbor</code> file in
 * a directory prior to indexing.
 */
public class CarEntityCollection extends DocumentCollection<CarEntityCollection.Document> {

  public CarEntityCollection(Path path) {
    this.path = path;
    this.allowedFileSuffix = new HashSet<>(Arrays.asList(".cbor"));
  }

  @Override
  public FileSegment<CarEntityCollection.Document> createFileSegment(Path p) throws IOException {
    return new Segment(p);
  }

  /**
   * An individual file in {@code CarCollection}.
   */
  public static class Segment extends FileSegment<CarEntityCollection.Document> {
    private final FileInputStream stream;
    private final Iterator<Data.Page> iter;

    public Segment(Path path) throws IOException {
      super(path);
      stream = new FileInputStream(new File(path.toString()));
      iter = DeserializeData.iterableAnnotations(stream).iterator();
    }

    @Override
    public void readNext() {
      System.setProperty("file.encoding", "UTF-8");
      Data.Page p;
      p = iter.next();
      StringBuilder buffer = new StringBuilder();
      for (Data.PageSkeleton skel : p.getSkeleton()) {
        recurseArticle(skel, buffer);
      }
      bufferedRecord = new Document(p.getPageId(), buffer.toString());
      if (!iter.hasNext()) {
        atEOF = true;
      }
    }
  }

  private static void recurseArticle(Data.PageSkeleton skel, StringBuilder buffer) {
    if (skel instanceof Data.Section) {
      final Data.Section section = (Data.Section) skel;
      buffer.append(System.lineSeparator());
      for (Data.PageSkeleton child : section.getChildren()) {
        recurseArticle(child, buffer);
      }
    } else if (skel instanceof Data.Para) {
      Data.Para para = (Data.Para) skel;
      Data.Paragraph paragraph = para.getParagraph();
      buffer.append(contentFromParagraph(paragraph));
    } else if (skel instanceof Data.Image) {
      Data.Image image = (Data.Image) skel;
      for (Data.PageSkeleton child: image.getCaptionSkel()) {
        recurseArticle(child, buffer);
      }
    } else if (skel instanceof Data.ListItem) {
      Data.ListItem item = (Data.ListItem) skel;
      Data.Paragraph paragraph = item.getBodyParagraph();
      buffer.append(contentFromParagraph(paragraph));
    } else {
      throw new UnsupportedOperationException("not known skel type " + skel.getClass().getTypeName() + "; " + skel);
    }
  }

  private static String contentFromParagraph(Data.Paragraph paragraph) {
    StringBuilder buffer = new StringBuilder();
    for (Data.ParaBody body : paragraph.getBodies()) {
      if (body instanceof Data.ParaLink) {
        Data.ParaLink link = (Data.ParaLink) body;
        buffer.append(link.getAnchorText());
      } else if (body instanceof Data.ParaText) {
        buffer.append(((Data.ParaText) body).getText());
      } else {
        throw new UnsupportedOperationException("not known body " + body);
      }
    }
    return buffer.toString();
  }
  /**
   * A document from a collection for the TREC Complex Answer Retrieval (CAR) Track.
   * The paraID serves as the id.
   * See <a href="http://trec-car.cs.unh.edu/datareleases/">this reference</a> for details.
   */
  public static class Document extends SourceDocument {
    private final String pageID;
    private final String page;

    public Document(String pageID, String page) {
      this.pageID = pageID;
      this.page = page;
    }

    @Override
    public String id() {
      return pageID;
    }

    @Override
    public String content() {
      return page;
    }

    @Override
    public boolean indexable() {
      return true;
    }
  }
}
